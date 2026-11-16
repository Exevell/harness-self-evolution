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
- **常数等价检测**：模型 RMSE vs 常数 RMSE
- **分轴承 bias 检测**：B 题最大绝对 bias < 500

### 3. 三项最优策略与实测结果

| 任务 | 策略 | 官方基线 → 本系统实测 | 闸门结论 |
|------|------|----------------------|----------|
| **A（气体识别）** | 类平衡加权 + 高斯增强 + 余弦退火 | Macro-F1 0.7140 → **0.7412**（+3.8%） | ✅ 通过，已上线 |
| **B（轴承 RUL）** | 向训练集中位数收缩（a 由大模型在 0.3/0.45/0.6 间决策） | RMSE 681.16 → **458.48 / 467.64 / 486.06** | ❌ 0/3 通过，全部回滚 |
| **C（发动机 RUL）** | 长寿命加权（rul-150x3）+ 300ep | RMSE 52.87 → **92.86 / 262.45 / 101.12** | ❌ 0/3 通过，全部回滚 |

> B 题分数虽比基线低 32.7%，但预测分布坍缩（IQR 仅为真值的 8%～16%）、后两轮与可部署常数 470.86 统计等价，闸门判定不可部署，最终交付基线检查点。C 题策略全部劣于常数 79.41。**这些是真实结果，不做修饰。** 详见 `docs/technical_report.md` 第 3～5 节。

### 4. 完整合规性
✅ 无人工干预自主运行  
✅ 完整实验记录（SwanLab 集成）  
✅ ≥3 次"读历史—做实验—依据结果决策"记录  
✅ 受控 RSI（退化检查三道闸门）  
✅ 可复现（所有超参数和随机种子记录）  

## 快速开始

### 环境要求

```bash
# Python 3.12.4（必须）
C:\Users\1\anaconda3\python.exe -V

# NumPy 1.26.4（必须）
C:\Users\1\anaconda3\python.exe -c "import numpy; print(numpy.__version__)"

# 安装依赖
C:\Users\1\anaconda3\python.exe -m pip install -r requirements.txt
```

### 一键运行

```bash
# B 题自进化（10 轮）
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task B \
  --rounds 10 \
  --base-dir C:\Users\1\Downloads \
  --output-dir submission_runs/B

# A 题自进化
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task A \
  --rounds 10

# C 题自进化
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task C \
  --rounds 10
```

### 上传到 SwanLab

```bash
C:\Users\1\anaconda3\python.exe upload_to_swanlab.py \
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
3. 等价纯常数 RMSE（模型 RMSE > 常数 RMSE × 0.95 为等价）

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
诊断（当前镜像实测）：跨批次 LDA 准确率 **0.4043** **低于**随机基线 **0.4277**，最大类质心偏移 **9.90**，判定 `covariate_shift / severe`——决策边界完全不可迁移。因此采用逆频率类平衡加权 + 高斯噪声增强（σ=0.5）+ 余弦退火。实测 Macro-F1 0.7140 → **0.7412**（+3.8%），通过闸门；但分类别 F1 中有一个类只有 0.0531，改进不均匀。

**B 题（轴承 RUL）**
诊断（当前镜像实测）：训练集寿命中位数 **758.5**；存在寿命空档 **532 → 2495** 分钟，验证集 Bearing3_4（真值 1514）正落其中；分轴承 MSE 占比 39.9% / 44.5% / 15.6%，最大 44.5% **低于** 80% 主导阈值，故诊断输出 `unknown / none`。收缩类策略（把预测向训练集中位数收缩）能把 RMSE 从官方基线 681.16 压到 458.48（−32.7%），但**代价是预测分布坍缩**（IQR 67 vs 真值 839），闸门据此拒绝上线。

**C 题（发动机 RUL）**
诊断（当前镜像实测）：长寿命区间（RUL>150）系统性低估 **−224.2** 循环，该区间样本占比 **25.3%**，判定 `long_life_undershoot / moderate`。策略池采用 rul-150x3 加权 + 300 epoch 续训，实测 RMSE 92.86 / 262.45 / 101.12，全部劣于可部署常数 79.41，全部回滚。根因是策略池时间尺度与基线不匹配（基线自身只训 15 epoch），已列为已知局限。

### 方法论教训

1. **单次 train/val 划分会给出与 LOBO 完全相反的结论**  
   → 按整根轴承划分的任务必须用 LOBO

2. **"按验证集挑最优轮次"是隐式早停，数字系统性乐观**  
   → 所有报告数字必须标注 `finalEpVal`（可部署）还是 `bestEpVal`（乐观）

3. **指标变好不等于方法有效，要检查是否退化成常数**  
   → 三道闸门：预测 IQR vs 真值 IQR、分轴承 bias、等价纯常数 RMSE

4. **pooled LOBO 会被巨型轴承污染**  
   → 按寿命分组报告（short/mid/long），并同时看 macro 与 worst-bearing

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
✅ **≥3 次读历史决策**：`ExperimentManager.read_history_and_decide()` 强制前 3 轮探索后读历史决策  
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

| 任务 | 官方基线 | 本系统实测 | 相对基线 | 闸门 |
|------|----------|------------|----------|------|
| A（气体识别） | Macro-F1 0.7140 | **0.7412** | +3.8% | ✅ 上线 |
| B（轴承 RUL） | RMSE 681.16 | 458.48 / 467.64 / 486.06 | −32.7% / −31.4% / −28.6% | ❌ 0/3，全部回滚 |
| C（发动机 RUL） | RMSE 52.87 | 92.86 / 262.45 / 101.12 | 未达标 | ❌ 0/3，全部回滚 |

数字来源：官方基线取各任务包 `training.json` 的 `validation_after`；实测取 `submission_runs/{A,B,C}/experiment_history.jsonl`。早期 README 曾写 0.7519 / 452.64 / 50.54 / NASA 3.448e9，这些在当前镜像无法复现，已撤回。

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
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task B \
  --rounds 10 \
  --resume
```

## 版本信息

- **版本**: 1.0.0
- **提交日期**: 2026-11-16
- **Git 提交**: （待填写）

## 技术支持

- **赛题文档**: `cpu-*/TASK.md`, `cpu-*/TRAINING_GUIDE.md`
- **Harness 文档**: `harness/README.md`
- **分析报告**: `progress-report-工业模型自进化赛题.md`
