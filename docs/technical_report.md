# 技术报告：面向工业制造的两层自进化 Harness

**版本**：1.0.0（2026-11-17 修订）
**镜像**：`harness:1.0.0`，linux/amd64，单层，入口 `/app/run.sh`
**基础模型**：组委会提供的 CPU NumPy MLP（A 4326 参数 / B 865 / C 833），架构不变，仅续训权重
**策略决策模型**：PyTRIO `Qwen/Qwen3.5-4B`（OpenAI 兼容端点）

---

## 0. 摘要

本系统实现一个**受控递归自改进（RSI）闭环**：漂移监测 → 诊断归因 → 策略选择 → 扩展训练 → 三道退化闸门评估 → 上线或回滚。策略选择由大模型（PyTRIO `Qwen/Qwen3.5-4B`）在**受约束的动作空间**内完成，闸门拥有最终否决权。

三个工业任务在**当前镜像**上的实测结果（全部可在本仓库的 `submission_runs/` 与 SwanLab 中复现）：

| 任务 | 指标 | 官方基线 | 本系统实测 | 相对基线 | 闸门结论 |
|------|------|----------|------------|----------|----------|
| A 气体识别 | Macro-F1 | 0.7140 | **0.7412** | **+3.8%** | ✅ 通过（round-01） |
| B 轴承 RUL | RMSE | 681.16 | 458.48 / 467.64 / 486.06 | −32.7% / −31.4% / −28.6% | ❌ 0/3 通过，全部回滚 |
| C 发动机 RUL | RMSE | 52.87 | 92.86 / 262.45 / 101.12 | 劣于基线 | ❌ 0/3 通过，全部回滚 |

**必须说清的一点**：只有 A 题真正"上线"了改进。B 题的三轮数值虽然都远低于官方基线 681.16，但三道闸门判定其预测分布已经坍缩到常数附近（预测 IQR 只有真值的 8%～16%，第 2、3 轮更是与"部署常数 470.86"统计等价），因此系统**主动拒绝上线**。C 题的三条策略全部劣于可部署常数 79.41，同样被拒绝。

这不是调参失败，而是闸门在起作用：**在本系统里，"分数变好"不等于"模型变好"**，闸门宁可保留基线检查点，也不接受一个靠输出常数骗分数的策略。第 3 节给出每道闸门在真实轮次上的触发数值。

---

## 1. 系统架构

### 1.1 目录结构

```
/app
├── run.sh                    # 入口：解析任务包、注入环境变量、调用 run_harness.py
├── run_harness.py            # 单入口 CLI（--task A|B|C --rounds N）
├── upload_to_swanlab.py      # 把 experiment_history.jsonl 上传到 SwanLab（可选）
├── harness/
│   ├── self_evolution.py     # 核心：6 步闭环、策略池、三道闸门、回滚
│   ├── llm_advisor.py        # 大模型策略顾问（PyTRIO / OpenAI 兼容）
│   └── ...
├── shared/train_ext.py       # 扩展训练器（加权/增强/退火/目标变换）
└── tasks/{A,B,C}/baselines/  # 赛题基线代码（model.py / data.py / train.py / predict.py）
```

### 1.2 六步闭环

| 步骤 | 动作 | 产物 |
|------|------|------|
| 1 | 加载检查点与训练/验证数据 | `current_checkpoint` |
| 2 | 漂移监测（仅用训练集特征与结构信息） | `monitor_result` |
| 3 | 诊断归因（四类根因：协变量漂移 / 主导轴承 / 长寿命低估 / 无异常） | `diagnosis` |
| 4 | 策略选择（大模型在受约束动作空间内决策，失败则启发式） | `strategy` + `llm` |
| 5 | 扩展训练（从当前检查点续训，产物写新检查点） | `candidate_checkpoint` |
| 6 | 三道退化闸门评估 → 上线或回滚 | `degradation_checks` + `passed` |

每轮把上述全部字段写入 `experiment_history.jsonl`，一轮一条 JSON，可逐字段追溯。

### 1.3 大模型不可用时的行为

`HARNESS_LLM=0` 关闭大模型；Key 缺失、超时、返回非法 JSON 时，`llm_advisor` 一律返回 `llm.source="heuristic"` 的兜底决策，**流程不中断**。选中的策略仍要过闸门，因此兜底不会带来未经验证的上线。

---

## 2. 大模型决策层（PyTRIO）

### 2.1 接口契约

| 项 | 值 |
|----|----|
| SDK / 协议 | OpenAI 兼容 Chat Completions（`POST /api/openai/v1/chat/completions`） |
| 端点 | `https://pytrio.com/api/openai/v1`（可用 `PYTRIO_BASE_URL` 覆盖） |
| 鉴权 | `Authorization: Bearer <PYTRIO_API_KEY>` |
| 可用模型 | `Qwen/Qwen3.5-4B`（实测可用）；`Qwen/Qwen3.8-27B` 返回 `HTTP 404 no available actor`，不可用 |
| Key 读取顺序 | `PYTRIO_API_KEY` → `TRIO_API_KEY` → `OPENAI_API_KEY` → `PYTRIO_API_KEY_FILE` → `/data/pytrio_api_key` |

### 2.2 一个必须记录的工程结论：关闭 thinking

`Qwen3.5-4B` 是推理型模型，默认会先产出大量 reasoning token。实测证据：`max_tokens` 设为 512 / 2048 / 4096 / 8192 时，`completion_tokens_details.reasoning_tokens` 均被吃满，`finish_reason=length`、`content=null`，即**回答永远为空**。

逐项试错的结果：

| 写法 | 结果 |
|------|------|
| 加大 `max_tokens`（512→8192） | ❌ 仍被 reasoning 吃光 |
| `chat_template_kwargs: {enable_thinking: false}` | ❌ HTTP 400 |
| `reasoning_effort: "none"` | ❌ HTTP 422 |
| **顶层 `enable_thinking: false`** | ✅ `reasoning_tokens=0`、`finish_reason=stop`、返回干净 JSON |

因此请求体固定包含 `"enable_thinking": false`。关掉思考后单次决策耗时 0.8～1.9 秒（实测三轮分别为 1.2s / 0.84s / 1.89s），prompt 约 1285 token，completion 约 104 token。

### 2.3 受约束的动作空间

模型**不能**自由生成代码或任意超参。它只能：

1. 从策略白名单里选一个策略名；
2. 对白名单内的超参给 `param_overrides`，服务端按 `PARAM_BOUNDS` 逐项夹取（越界值被夹到边界，不报错）；
3. 输出结构固定为 `{"strategy", "param_overrides", "reason", "code_notes"}`。

提示词上下文包含：基线 `model.py` 的代码摘录（≤900 字符）、`configs/*.json`、本轮诊断结果、最近 3 轮历史（策略名 + 指标 + 是否通过）。要求只输出 JSON。

### 2.4 真实决策样例（B 题 round-01）

```json
{"source": "llm", "strategy": "B_shrink030", "param_overrides": {"shrink_a": 0.3},
 "reason": "诊断显示无显著主导轴承且当前策略为B_shrink045，降低shrink_a至0.3可增强模型对非主导样本的拟合能力以改善RMSE",
 "code_notes": "基线代码使用Adam优化器且默认学习率为0.003，当前候选策略均保持此设置，仅调整shrink_a参数以响应诊断结果",
 "model": "Qwen/Qwen3.5-4B", "latency_s": 1.2,
 "usage": {"prompt_tokens": 1285, "completion_tokens": 104},
 "finish_reason": "stop"}
```

三轮的 `llm.source` 全部是 `llm`，没有一次落到启发式兜底。SwanLab 记录：<https://swanlab.cn/@070228Ggx/harness-self-evolution/runs/45z9n9nn>

---

## 3. 三道退化闸门

| 闸门 | 检测目标 | 判据 | 设计理由 |
|------|----------|------|----------|
| IQR 坍缩 | 预测分布是否退化成常数 | `pred_iqr / true_iqr < 0.1` | 分布坍缩后 RMSE 只反映"碰巧接近中位数"，不代表学到映射 |
| 常数等价 | 是否与"可部署常数"统计等价 | `rmse_model > rmse_const × 0.98` 判为等价 | 与常数打平就没有部署价值 |
| 分轴承 bias | B 题单轴承系统性偏移 | `max_abs_bias > 500` 分钟 | 500 分钟是验证集 RMSE 量级 |

**"可部署常数"的口径**：取训练集标签中位数（B 题 758.5），在验证集上计算 RMSE＝**470.86**。不使用验证集标签中位数（465.73，预言机常数）作为比较基准，因为那需要验证集标签，赛规不允许——该值仅作为 `rmse_const_oracle` 字段记录，不参与判定。

### 3.1 真实触发记录

| 轮次 | 策略 | RMSE | 预测 IQR / 真值 IQR | 常数等价 | 分轴承 max bias | 结论 |
|------|------|------|---------------------|----------|-----------------|------|
| B round-01 | shrink_a=0.30 | 458.48 | 67.18 / 839 = **0.080** | 否 | 581.49 | ❌ IQR 坍缩 |
| B round-02 | shrink_a=0.45 | 467.64 | 100.77 / 839 = 0.120 | **是**（470.86×0.98=461.4） | 505.23 | ❌ 与常数等价 |
| B round-03 | shrink_a=0.60 | 486.06 | 134.36 / 839 = 0.160 | **是** | 440.98 | ❌ 与常数等价 |
| C round-01 | rul150x3, 300ep | 92.86 | 0.051 / 111 = **0.00046** | **是**（79.41×0.98=77.8） | — | ❌ 双向触发 |
| C round-02 | rul125x2, 300ep | 262.45 | 40.28 / 111 = 0.363 | **是** | — | ❌ |
| C round-03 | uniform, 300ep | 101.12 | 0.053 / 111 = **0.00047** | **是** | — | ❌ |

这张表是本系统最有说服力的部分：**B 题第 1 轮 458.48 已经比官方基线 681.16 好 32.7%，闸门仍然拒绝上线**，因为它看清了这个分数的来源——把预测向中位数压缩，让 RMSE 靠近"常数 RMSE"，同时把预测分布压平到真值 IQR 的 8%。这正是自进化系统最容易走上的作弊路径，也是赛题要求"退化检查"的原因。

---

## 4. 三题实测明细

### 4.1 A 题（气体识别，分类）

- 诊断：`covariate_shift` / `severe`；跨批次 LDA 准确率 **0.4043** < 随机基线 **0.4277**，最大质心偏移 **9.90**
- 策略：`A_invfreq_gauss05_cos`（逆频率类平衡加权 + 高斯增强 σ=0.5 + 余弦退火）
- 实测：Macro-F1 **0.7412**，Accuracy **0.8955**，通过闸门
- 官方基线：Macro-F1 0.7140，Accuracy 0.8786 → **+3.8%**
- 分类别 F1：`[0.7140, 0.9699, 1.0000, 0.0531, 0.9962, 0.7143]`

**诚实提示**：第 4 个类的 F1 只有 0.0531，说明类平衡加权救不回一个在验证批次里几乎无法区分的类别。宏平均被其余 5 个高分类别拉起来，因此 +3.8% 是真实的、但不是均匀的改进。

### 4.2 B 题（轴承 RUL，回归）

监测到的结构事实（只用训练集信息）：训练集寿命中位数 **758.5** 分钟；存在寿命空档 **532 → 2495** 分钟（跨度 1963）；验证集 `Bearing3_4`（真值 1514 分钟）正落在这个空档内。验证集分轴承 MSE 占比 `Bearing1_4 39.9% / Bearing2_4 44.5% / Bearing3_4 15.6%`，最大占比 44.5% **低于** 80% 的主导阈值，因此诊断输出 `unknown / none`，动作是 `monitor_only`。

三轮全部由大模型决策，全部被回滚：

| 轮次 | 策略 | 决策理由（模型原文节选） | RMSE | 结果 |
|------|------|--------------------------|------|------|
| round-01 | shrink_a=0.30 | "降低 shrink_a 至 0.3 可增强模型对非主导样本的拟合能力" | 458.48 | 回滚（IQR 0.080） |
| round-02 | 不改超参 | "无需调整超参即可维持监控状态" | 467.64 | 回滚（常数等价） |
| round-03 | shrink_a=0.60 | "历史策略 B_shrink045 表现更差，因此选择 B_shrink060 以扩大收缩范围" | 486.06 | 回滚（常数等价） |

系统最终保留**原始基线检查点**，没有上线任何 shrink 策略。相对官方基线 681.16，第 1 轮的 458.48 是 −32.7%，但系统判定它不可部署——这个判断可以被任何人用 `degradation_checks.iqr_collapse` 字段复核。

### 4.3 C 题（发动机 RUL，回归）

- 诊断：`long_life_undershoot` / `moderate`；长寿命区间低估 **−224.2** 循环，该区间样本占比 **25.3%**
- 可部署常数 RMSE **79.41**（训练集中位数 114.0）；预言机常数 79.87（108.0）；真值 IQR 111
- 三轮：`C_rul150x3_ep300` 92.86 → `C_rul125x2_ep300` 262.45 → `C_uniform_ep300` 101.12，全部劣于常数，全部回滚
- 官方基线 RMSE **52.87**（MAE 38.90）

**这是本系统的已知短板，必须写明**：C 题的策略池是在基线检查点上以 300 epoch 重训，训练量远超基线自身的 15 epoch / 6630 step，导致模型偏移到比基线差 1.7～5 倍。闸门正确拦下了它（没有污染最终产物），但策略池本身对 C 题无效。下一步的修复方向是给策略池加入"零改动基线"与"小 epoch 续训"两个安全选项，并让大模型看到基线的 `training.json`（含 15 epoch 这一事实）。

---

## 5. 数据口径（哪些数字可以查、查到哪里）

本报告的数字只有两个来源，其余一律撤回：

| 来源 | 内容 | 位置 |
|------|------|------|
| 赛题官方 | A Macro-F1 0.7140 / Acc 0.8786；B RMSE 681.16 / MAE 563.11；C RMSE 52.87 / MAE 38.90 | 各任务包 `training.json` 的 `validation_after` |
| 本系统实测 | 第 0、3、4 节全部数值 | `submission_runs/{A,B,C}/experiment_history.jsonl`、`B题三轮-真实LLM-7.log`、SwanLab run `45z9n9nn` |

**已撤回的旧数字**：本仓库早期文档出现过 `0.7519`、`452.64`、`50.54`、`NASA 3.448e9`、`NASA 5.823e9`、`452.64 → 385.40` 等表述，声称 A/B/C 三题分别提升 +5.3% / −33.5% / −40.8%。这些数值来自探索阶段的离线脚本，**在当前提交的镜像与代码中无法复现**，因此已从技术报告、设计论证、运行说明、演示材料、合规声明中全部删除。若评委在旧材料里看到过它们，以本文件为准。

---

## 6. 复现

```bash
# 镜像内（平台挂载 /data/train、/data/validation、/data/checkpoint）
/app/run.sh                      # 读 TASK / ROUNDS 环境变量
TASK=A ROUNDS=1 /app/run.sh

# 本机（Windows/Linux 均可，需赛题包目录）
export HARNESS_BASELINES=/path/to/tasks/B/baselines
export PYTRIO_API_KEY=<key>      # 不设则自动降级为启发式
python run_harness.py --task B --rounds 3 \
  --base-dir /path/to/packages --output-dir ./submission_runs/B
```

产物：`experiment_history.jsonl`（逐轮完整记录）、`round-NN/{checkpoint.npz,training.json,history.jsonl}`。最终可用检查点是最后一个通过闸门的 `round-NN/checkpoint.npz`；若全部回滚，则仍是输入基线检查点。上传 SwanLab：

```bash
python upload_to_swanlab.py --history submission_runs/B/experiment_history.jsonl \
  --project harness-self-evolution
```

---

## 7. 已知局限

1. **C 题策略池无效**：如 4.3 所述，300 epoch 重训把模型带离基线。闸门拦住了，但策略本身需要重做。
2. **B 题没有可部署策略**：收缩类策略全部落进"分布坍缩"区间。真正可行的方向是在空档区间做外推（例如按轴承寿命插值），而不是全局收缩。
3. **A 题改进不均**：单个类别 F1 0.0531。
4. **大模型只被约束在参数层**：`code_notes` 字段记录了模型对代码的观察，但没有让它真正改动代码——这是刻意的安全边界，避免不可审计的自改代码。
5. **闸门阈值是人工设定的常量**（0.1 / 0.98 / 500），未做自适应。
