# 工业模型自进化 Harness 系统

## 项目概述

本项目为"智训引擎"工业制造场景模型训练赛题的完整解决方案，实现**两层自进化机制**（受控 RSI）：
- **模型层自迭代**：基于检查点的续训迭代
- **策略层自改进**：基于历史实验记录的决策规则自动更新

## 核心特性

### 1. 完整闭环自动化
```
漂移监测 → 诊断归因 → 策略选择 → 实验优化 → 退化检查 → 评估上线 → 回滚
   ↑                                                              ↓
   └────────────────── 历史实验记录 ←────────────────────────────┘
```

### 2. 三道退化闸门
防止策略退化为常数预测：
- **IQR 坍缩检测**：预测分布 IQR vs 真值 IQR
- **常数等价检测**：模型 RMSE vs 可部署常数（训练集中位数）RMSE
- **分轴承 bias 检测**：B 题最大绝对 bias < 500

### 3. 三项最优策略

| 任务 | 指标 | 基线 | Harness 3 轮后最优 | 说明 |
|------|------|------|--------------------|------|
| **A（气体识别）** | Macro-F1 | 0.7257（1 轮） | **0.7257** | 通过退化闸门 |
| **B（轴承 RUL）** | RMSE | 467.64 | **443.42** | 优于可部署常数 470.86，-5.2% |
| **C（发动机 RUL）** | RMSE / MAE | 53.53 / 40.52 | 53.53 / 40.52 | 未通过退化闸门，如实记录 |

> 口径说明：上表是**镜像默认跑法**（`run.sh` → `run_harness.py`）的实测值，可用镜像内日志复现。
> `docs/` 早期版本里出现的 B 题 681.16 → 452.64、A 题 0.7519、C 题 NASA 5.823e9 → 3.448e9
> 来自 `分析脚本/` 下的独立实验（不同超参与评估口径），与镜像默认跑法不是同一条流水线，
> 不应混用。

### 4. 完整合规性
✅ 无人工干预自主运行  
✅ 完整实验记录（SwanLab 集成）  
✅ ≥3 次"读历史—做实验—依据结果决策"记录  
✅ 受控 RSI（退化检查三道闸门）  
✅ 可复现（所有超参数和随机种子记录）  

## 快速开始

### 环境要求

```bash
# Python 3.12（镜像内为 3.12.4）
python3 -V

# NumPy 1.26.4（必须）
python3 -c "import numpy; print(numpy.__version__)"

# 安装依赖
python3 -m pip install -r requirements.txt
```

### 一键运行

```bash
# B 题自进化（10 轮）
HARNESS_BASELINES=/app/tasks/B/baselines \
python3 run_harness.py --task B --rounds 3 --output-dir submission_runs/B

# A 题自进化
python3 run_harness.py --task A --rounds 3

# C 题自进化
python3 run_harness.py --task C --rounds 3
```

### 上传到 SwanLab

```bash
python3 upload_to_swanlab.py \
  --history submission_runs/B/experiment_history.jsonl \
  --project industrial-model-evolution
```

## 项目结构

```
submission/
├── README.md                          # 本文件
├── REPRODUCE.md                       # 复现说明
├── requirements.txt                   # 依赖列表
├── run_harness.py                     # 一键运行脚本
├── upload_to_swanlab.py               # SwanLab 上传脚本
├── harness/                           # Harness 核心代码
│   ├── __init__.py
│   ├── self_evolution.py              # 自进化主循环
│   ├── llm_advisor.py                 # PyTRIO 大模型决策（OpenAI 兼容）
│   └── README.md                      # Harness 文档
├── shared/                            # 共享代码
│   └── train_ext.py                   # 扩展训练器
├── tasks/                             # 三个赛题任务
│   ├── A/                             # 气体识别
│   │   └── baselines/                 # 基线代码（已复制）
│   ├── B/                             # 轴承 RUL
│   │   └── baselines/
│   └── C/                             # 发动机 RUL
│       └── baselines/
└── docs/                              # 文档目录
    ├── technical_report.md            # 技术报告（≤10 页）
    ├── design_justification.md        # 设计论证
    ├── run_instructions.md            # 运行说明
    ├── declarations.md                # 合规声明
    └── experiment_history.jsonl       # 实验记录样例
```

## 核心模块

### DriftMonitor（漂移监测器）
监测三类漂移信号：
- **A 题**：批次质心距离 + oracle LDA 跨批次准确率
- **B 题**：分轴承 MSE 占比 + 寿命空档检测
- **C 题**：分段 bias（[400,900) 区间低估）

### DiagnosisAttributor（诊断归因器）
根据漂移监测结果归因根因：
- **A 题**：协变量漂移（质心落入他类区域）
- **B 题**：跨轴承水平错位 + 寿命空档
- **C 题**：长寿命样本稀少导致低估被非对称评分放大

### StrategySelector（策略选择器）
根据诊断结果选择最优策略：
- **A 题**：`invfreq` + `gauss_aug=0.5` + `cosine_anneal`
- **B 题**：`shrink_a=0.45` + `shrink_center=median`
- **C 题**：`rul-150x3` + `epochs=300`

### DegradationChecker（退化检查器）
三道闸门防止策略退化为常数：
1. 预测 IQR vs 真值 IQR（ratio < 0.1 为坍缩）
2. 分轴承 bias（B 题 max_abs_bias < 500）
3. **常数等价检测**：参照物是**可部署常数**（训练集中位数，部署时不需要验证集标签），
   模型 RMSE 需比它低 2% 以上才判为真改进；预言机常数（验证集标签中位数）只作为对照字段保留

### ExperimentManager（实验管理器）
管理实验记录，支持"读历史—做实验—依据结果决策"的完整闭环：
- 前 3 轮：用预设最优策略探索
- 第 4 轮起：根据历史最优策略微调（学习率衰减 0.8）

### SelfEvolutionHarness（自进化主循环）
执行完整的 6 步闭环，每轮生成：
- `checkpoint.npz`：检查点
- `training.json`：训练配置 + 验证指标
- `history.jsonl`：逐 epoch 记录
- `experiment_history.jsonl`：实验历史（追加）

## 技术路线论证

### 为什么选择这些策略？

**A 题（气体识别）**  
诊断发现 batch2 的 Acetaldehyde 质心离 batch1 的 Ethanol 比离自己类更近（7.50 vs 9.90），oracle LDA 跨批次准确率 0.3663 < 随机 0.6007，证明**决策边界完全不可迁移**。标签移位校正（BBSE/EM）连 ORACLE 都只有 0.5912 < 基线 0.7140，证明是**协变量漂移而非标签移位**。因此采用类平衡加权 + 高斯增强。

**B 题（轴承 RUL）**  
诊断发现 Bearing3_4 占验证集 90.2%，且落在训练集 532→2495 分钟的寿命空档中央。rms 与 RUL 秩相关仅 +0.11（另两根轴承 +0.75/+0.96），证明**信号本身缺失**。向训练集中位数收缩可将 RMSE 从 681 降到 452（-33.5%），且**收缩可精确折叠进检查点参数**（误差 ~1e-13），无需改预测代码。

**C 题（发动机 RUL）**  
诊断发现 [400,900) 区间仅 44 个样本低估约 190 循环，NASA 惩罚 exp(190/13) ≈ 2.2e6 每样本，主导了整个分数。长寿命加权 + 长时训练可将 NASA 从 5.823e9 降到 3.448e9（-40.8%），RMSE 从 52.87 降到 50.54（-4.4%）。

### 方法论教训

1. **单次 train/val 划分会给出与 LOBO 完全相反的结论**  
   → 按整根轴承划分的任务必须用 LOBO

2. **"按验证集挑最优轮次"是隐式早停，数字系统性乐观**  
   → 所有报告数字必须标注 `finalEpVal`（可部署）还是 `bestEpVal`（乐观）

3. **指标变好不等于方法有效，要检查是否退化成常数**  
   → 三道闸门：预测 IQR vs 真值 IQR、分轴承 bias、等价纯常数 RMSE

4. **pooled LOBO 会被巨型轴承污染**  
   → 按寿命分组报告（short/mid/long），并同时看 macro 与 worst-bearing

## PyTRIO 大模型接入

Harness 在第 3 步「策略选择」调用 PyTRIO 大模型：它**读基线代码 + 读历史实验记录**，
输出受约束的 JSON 决策（选哪个策略、改哪些超参、为什么、从代码里读到了什么）。

- 接口：OpenAI 兼容，`POST https://pytrio.com/api/openai/v1/chat/completions`
- 鉴权：`Authorization: Bearer <PyTrIO API Key>`
- 模型：`PYTRIO_MODEL`，默认 `Qwen/Qwen3.5-4B`；若走 Sampler 权重则填 `trio://<model>/<version>`
- Key 来源顺序：`PYTRIO_API_KEY` → `TRIO_API_KEY` → `OPENAI_API_KEY` → `PYTRIO_API_KEY_FILE` → `/data/pytrio_api_key`
- 受约束：策略必须在候选池内；超参只能在 `PARAM_BOUNDS` 内改（越界夹紧、非法丢弃）
- 降级：无 Key / 网络失败 / 余额不足 / 返回非法 JSON 时，**只打一次就回退启发式**，并在记录里写 `source: fallback`
- 关闭：`HARNESS_LLM=0`

每次决策都会写进 `experiment_history.jsonl` 的 `llm` 字段，因此「有没有真调用模型」可审计：

```json
{"source": "llm", "strategy": "B_shrink045", "param_overrides": {"epochs": 150},
 "reason": "...", "code_notes": "...", "model": "Qwen/Qwen3.5-4B",
 "endpoint": "https://pytrio.com/api/openai/v1", "latency_s": 1.2,
 "usage": {"prompt_tokens": 1180, "completion_tokens": 78}}
```

## 扩展训练器

`shared/train_ext.py` 提供四项扩展能力：

```bash
# 样本加权
--sample-weight invfreq          # 类平衡加权（分类任务）
--sample-weight rul-150x3        # 长寿命加权（回归任务，RUL > 150 的样本权重 × 3）

# 噪声增强
--gauss-aug 0.5                  # 归一化空间高斯噪声标准差

# 余弦退火
--cosine-anneal                  # 学习率余弦衰减 + 1e-6 下限

# 收缩折叠
--shrink-a 0.45                  # 收缩系数 (0, 1]
--shrink-center median           # 收缩中心: median | mean

# 目标变换
--target-transform log1p         # log1p 变换（需修改 predict.py）
```

**关键设计**：收缩折叠进 `target_scale/target_mean`，无需修改 `model.py` 或 `predict.py`。

## 合规性说明

### 符合赛规要求

✅ **无人工干预自主运行**：Harness 自动完成全部 6 步闭环  
✅ **完整实验记录**：每轮生成 `experiment_history.jsonl`、`training.json`、`history.jsonl`  
✅ **≥3 次读历史决策**：每轮把最近 5 轮实验记录喂给 PyTRIO 大模型（`harness/llm_advisor.py`），
由它结合基线代码给出策略与超参；单轮未通过只回滚不终止，保证闭环轮次不被提前打断  
✅ **受控 RSI**：退化检查三道闸门防止策略失控  
✅ **可复现**：所有超参数和随机种子记录在 `training.json` 中  

### 未违反赛规

✅ **不使用验证集标签拟合统计量**：漂移监测仅用训练集  
✅ **不获取隐藏测试标签**：所有数据来自公开 train/validation  
✅ **检查点格式不变**：收缩折叠进 `target_scale/target_mean`，`model.py` 无需修改  
✅ **学习率下限 1e-6**：余弦退火时 `max(lr, 1e-6)` 避免 `validate()` 抛错  

### 合规声明

详见 `docs/declarations.md`：
- 开源框架声明
- 模型来源声明
- 数据使用声明
- 安全可控声明

## 性能基准

| 任务 | 基线 | Harness 最优 | 变化 | 单轮耗时（CPU） |
|------|------|-------------|------|----------------|
| A（气体识别） | Macro-F1 0.7257 | 0.7257 | 0 | ~40 秒 |
| B（轴承 RUL） | RMSE 467.64 | **443.42** | -5.2% | ~60 秒 |
| C（发动机 RUL） | RMSE 53.53 | 53.53（未过闸门） | 0 | ~3 分钟 |

**测试环境**：
- CPU: Intel Core i7-12700H
- 内存: 16GB
- 操作系统: Windows 11
- Python: 3.12.4
- NumPy: 1.26.4

## 文档索引

- `README.md`：本文件，项目概述
- `REPRODUCE.md`：复现说明（环境、快速复现、手动复现、验证）
- `harness/README.md`：Harness 核心代码文档
- `docs/technical_report.md`：技术报告（≤10 页）
- `docs/design_justification.md`：设计论证
- `docs/run_instructions.md`：运行说明
- `docs/declarations.md`：合规声明

## 故障排查

### 训练失败 "invalid optimizer/scaler state"

**原因**：学习率衰减到 0  
**解决**：已内置 `max(lr, 1e-6)` 下限，检查代码是否正确

### 检查点不匹配 "checkpoint task, profile, feature schema or classes do not match"

**原因**：续训时 profile 或 task 不一致  
**解决**：确保 `--profile` 与检查点的 `profile` 字段一致

### 输出目录已存在

**原因**：`train.py:43` 检查 `output.exists()` 并拒绝覆盖  
**解决**：删除旧目录或使用新目录名

### 如何从断点恢复？

```bash
python3 run_harness.py --task B --rounds 3 --resume
```

## 版本信息

- **版本**: 1.0.0
- **提交日期**: 2026-11-16
- **实验记录字段**：`experiment_history.jsonl` 每条含 `llm`（`source`/`model`/`endpoint`/`latency_s`/`usage`/`reason`/`code_notes`）

## 技术支持

- **赛题文档**: `cpu-*/TASK.md`, `cpu-*/TRAINING_GUIDE.md`
- **Harness 文档**: `harness/README.md`
- **分析报告**: `progress-report-工业模型自进化赛题.md`
