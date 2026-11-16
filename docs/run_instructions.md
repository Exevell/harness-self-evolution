# 运行说明

## 1. 平台侧运行（镜像）

镜像 `harness:1.0.0`（linux/amd64，单层，`Entrypoint=/app/run.sh`）。平台注入环境变量后直接启动容器即可，无需任何人工操作。

| 环境变量 | 是否必需 | 默认值 | 说明 |
|----------|----------|--------|------|
| `TASK` | 是 | `B` | `A` / `B` / `C` |
| `ROUNDS` | 否 | `10` | 自进化轮数 |
| `CHECKPOINT` | 是 | — | 基线检查点 `checkpoint.npz` 绝对路径（平台挂载到 `/data/checkpoint`） |
| `TRAIN_JSONL` | 是 | — | 训练集 JSONL |
| `VAL_JSONL` | 是 | — | 验证集 JSONL |
| `OUTPUT_DIR` | 否 | `/tmp/harness_output` | 输出目录 |
| `BASELINES_DIR` | 否 | `/app/tasks/$TASK/baselines` | 含 `model.py` 的目录；镜像内已内置，正常无需改 |
| `PYTRIO_API_KEY` | 否 | — | 大模型 Key；不设则自动降级为启发式决策 |
| `PYTRIO_BASE_URL` | 否 | `https://pytrio.com/api/openai/v1` | 端点覆盖 |
| `PYTRIO_MODEL` | 否 | `Qwen/Qwen3.5-4B` | 模型名 |
| `PYTRIO_TIMEOUT` | 否 | `180` | 单次决策超时（秒） |
| `HARNESS_LLM` | 否 | 启用 | 设为 `0` 完全关闭大模型 |

启动方式二选一：

```bash
# A. 平台注入环境变量后直接跑入口
docker run --rm -v /host/data:/data -e TASK=B -e ROUNDS=10 harness:1.0.0

# B. 覆盖入口，手动指定路径
docker run --rm -v /host/data:/data \
  -e TASK=A -e ROUNDS=3 \
  -e CHECKPOINT=/data/checkpoint/checkpoint.npz \
  -e TRAIN_JSONL=/data/train/train.jsonl \
  -e VAL_JSONL=/data/validation/validation.jsonl \
  harness:1.0.0 /app/run.sh
```

`run.sh` 会依次校验检查点、训练集、验证集、`baselines/model.py` 是否存在，任一缺失即 `exit 1` 并打印明确原因，不会静默失败。

---

## 2. 本机运行（不经 Docker）

需要赛题包目录（含 `baselines/`、`public/`、`checkpoint.npz`）。

### 2.1 Windows（本团队实测环境）

```powershell
$root = 'D:\Documents\...\20261115-235804-4e9d4ddb-90e0-4a41-a17e-34cdc92482b0'
$sub  = "$root\submission"
Set-Location -LiteralPath $sub

$env:PYTHONIOENCODING = 'utf-8'
$env:HARNESS_BASELINES = "$sub\tasks\B\baselines"
$env:PYTRIO_API_KEY    = '<key>'      # 不设则走启发式
# $env:HARNESS_LLM     = '0'         # 想关掉大模型时打开

& 'C:\Users\1\anaconda3\python.exe' run_harness.py `
    --task B --rounds 3 `
    --base-dir 'C:\Users\1\Downloads' `
    --output-dir "$root\submission_runs\B"
```

> 注意：本机系统 `python` 可能是 WindowsApps 存根，必须用解释器绝对路径。

### 2.2 Linux / macOS

```bash
export HARNESS_BASELINES=$PWD/tasks/B/baselines
export PYTRIO_API_KEY=<key>
python3 run_harness.py --task B --rounds 3 \
  --base-dir /path/to/packages --output-dir ./submission_runs/B
```

---

## 3. 真实运行输出（节选）

### 3.1 A 题，1 轮（通过闸门）

```
======================================================================
自进化 Harness - 一键运行脚本
======================================================================
任务: A (气体识别)
轮数: 1
赛题包: C:\Users\1\Downloads\cpu-a-guide-20261110-v1
评估指标: macro_f1
======================================================================
[2/6] 漂移监测...
  跨批次 LDA 准确率: 0.4043 (随机基线: 0.4277)
[3/6] 策略选择...
  ✅ 大模型决策: A_invfreq_gauss05_cos
[6/6] 评估上线...
  ✅ 上线成功

📊 轮次 1/1 结果:
   Macro-F1: 0.7412
   通过: ✅
   策略: A_invfreq_gauss05_cos

通过轮数: 1
最优 Macro-F1: 0.7412
```

### 3.2 B 题，3 轮（全部回滚，全部由大模型决策）

```
[2/6] 漂移监测...
  最大 MSE 占比: 44.5% (阈值 80.0%) → 无主导轴承
[3/6] 策略选择...
  调用 PyTRIO 大模型（Qwen/Qwen3.5-4B @ https://pytrio.com/api/openai/v1）...
  ✅ 大模型决策: B_shrink030 {'shrink_a': 0.3} (1.2s)
     理由: 诊断显示无显著主导轴承且当前策略为B_shrink045，降低shrink_a至0.3可增强模型对非主导样本的拟合能力以改善RMSE
   RMSE: 458.48
  ⚠️ 预测分布坍缩 (IQR 67.18 / 真值 839, ratio 0.080)
   通过: ❌
  ⚠️ 轮次 1 未通过退化检查，已回滚，继续下一轮探索

  ✅ 大模型决策: B_shrink045 (不改超参) (0.84s)
   RMSE: 467.64
  ⚠️ 预测等价于常数 (RMSE=467.64 (可部署常数 470.86 / 预言机常数 465.73))
   通过: ❌

  ✅ 大模型决策: B_shrink060 (不改超参) (1.89s)
   RMSE: 486.06
  ⚠️ 预测等价于常数
   通过: ❌

通过轮数: 0
```

### 3.3 C 题，3 轮（全部回滚）

```
[3/6] 策略选择...
  策略: C_rul150x3_ep300
  ⚠️ 预测等价于常数 (RMSE=92.86 (可部署常数 79.41 / 预言机常数 79.87))
   RMSE: 92.86   通过: ❌
  策略: C_rul125x2_ep300
   RMSE: 262.45  通过: ❌
  策略: C_uniform_ep300
   RMSE: 101.12  通过: ❌
通过轮数: 0
```

三题的完整日志见仓库根目录 `A题-实测-1轮.log`、`B题三轮-真实LLM-7.log`、`C题-实测-3轮.log`。

---

## 4. 产物

| 文件 | 内容 |
|------|------|
| `experiment_history.jsonl` | 每轮一条 JSON：策略与全部超参、指标、诊断证据、监测原始值、三道闸门逐项数值、`passed`、`llm` 决策元信息 |
| `round-NN/checkpoint.npz` | 该轮训练产物。**只有通过闸门的轮次**成为下一轮的起点；全部回滚时下一轮仍从基线检查点开始 |
| `round-NN/training.json` | 该轮训练元信息（epochs、steps、学习率、目标变换参数等） |
| `round-NN/history.jsonl` | 该轮逐 epoch 训练/验证历史 |

> 系统**不**额外输出 `best_checkpoint.npz` 或 `summary.json`：最终可用检查点是"最后一个通过闸门的 `round-NN/checkpoint.npz`"，若全部回滚则就是输入基线检查点自身。这个设计是为了避免出现"另一个声称最优"的检查点，让产物只有一个解释。

单轮记录示例（已裁剪，真实字段名不变）：

```json
{"stage_id": "round-01",
 "strategy": {"name": "B_shrink030", "shrink_a": 0.3, "epochs": 120, "learning_rate": 0.01, "batch_size": 128},
 "validation_metric": 458.47844336937425,
 "passed": false,
 "degradation_checks": {"iqr_collapse": {"collapsed": true, "ratio": 0.08007084509844, "pred_iqr": 67.17943903759124, "true_iqr": 839.0},
                        "constant_equivalence": {"equivalent_to_constant": false, "rmse_model": 458.47844336937425, "rmse_const": 470.8582356845544},
                        "max_abs_bias": 581.4948473681497},
 "llm": {"source": "llm", "model": "Qwen/Qwen3.5-4B", "latency_s": 1.2,
         "reason": "…降低shrink_a至0.3可增强模型对非主导样本的拟合能力…"}}
```

---

## 5. 上传 SwanLab（可选）

SwanLab 只在**主流程之外**由人手动调用，镜像运行时不联网上传。

```bash
pip install swanlab          # 单独 venv，避免与主环境 protobuf 冲突
export SWANLAB_API_KEY=<key>
python upload_to_swanlab.py --history submission_runs/B/experiment_history.jsonl \
  --project harness-self-evolution
```

脚本只上传**数值型指标**（字符串字段如 `strategy_name` 会被 SwanLab 拒收，已提前剔除）。本系统最终记录：<https://swanlab.cn/@070228Ggx/harness-self-evolution/runs/45z9n9nn>

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `错误: 未找到 baselines 目录` | 镜像外运行且未设 `HARNESS_BASELINES` | 设 `HARNESS_BASELINES` 或 `BASELINES_DIR` |
| 决策全是 `heuristic` | 没给 Key，或 Key 无余额 | 检查 `PYTRIO_API_KEY`；确认账户有余额（409 报 `insufficient_quota`） |
| `content` 为空、`finish_reason=length` | 请求体缺 `enable_thinking: false` | 本镜像已内置该参数，不要自行删掉 |
| 轮次被大量回滚 | 属正常：闸门在起作用 | 读 `degradation_checks` 定位是哪道闸门 |
| 想复现无 LLM 的对照 | — | `HARNESS_LLM=0` |

---

## 7. 环境与依赖（镜像内实测）

| 组件 | 版本 |
|------|------|
| Python | 3.12.4 |
| NumPy | 1.26.4 |
| scikit-learn | 1.9.1 |
| SciPy | 1.17.1 |
| swanlab | 0.10.5（仅上传脚本使用，不在镜像主流程中） |
