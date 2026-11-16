# 工业模型自进化 Harness 系统

## 概述

本系统实现"智训引擎"赛题要求的**两层自进化机制**（受控 RSI）：
- **模型层自迭代**：基于检查点的续训迭代
- **策略层自改进**：基于历史实验记录的决策规则自动更新

## 核心架构

```
漂移监测 → 诊断归因 → 策略选择 → 实验优化 → 退化检查 → 评估上线 → 回滚
   ↑                                                              ↓
   └────────────────── 历史实验记录 ←────────────────────────────┘
```

## 模块说明

### 1. 漂移监测器 (DriftMonitor)

监测三类漂移信号：
- **A 题（气体识别）**：批次质心距离 + oracle LDA 跨批次准确率
- **B 题（轴承 RUL）**：分轴承 MSE 占比 + 寿命空档检测
- **C 题（发动机 RUL）**：分段 bias（[400,900) 区间低估被 NASA 非对称评分放大）

### 2. 诊断归因器 (DiagnosisAttributor)

根据漂移监测结果归因根因：
- **A 题**：协变量漂移（质心落入他类区域）vs 标签移位（先验变化）
- **B 题**：跨轴承水平错位 vs 形状误差
- **C 题**：长寿命样本稀少导致低估被非对称评分放大

### 3. 策略选择器 (StrategySelector)

根据诊断结果选择最优策略：
- **A 题**：类平衡加权 (`invfreq`) + 高斯增强 (`gauss_aug=0.5`) + 余弦退火
- **B 题**：向训练集中位数收缩 (`shrink_a=0.45`)
- **C 题**：长寿命样本加权 (`rul-150x3`) + 长时训练

### 4. 退化检查器 (DegradationChecker)

三道闸门防止策略退化为常数：
1. **预测 IQR vs 真值 IQR**：坍缩检测（ratio < 0.1）
2. **分轴承 bias**：B 题最大绝对 bias < 500
3. **等价纯常数 RMSE**：模型 RMSE 不应 > 常数 RMSE × 0.95

### 5. 实验管理器 (ExperimentManager)

管理实验记录，支持"读历史—做实验—依据结果决策"的完整闭环：
- 前 3 轮：用预设最优策略探索
- 第 4 轮起：根据历史最优策略微调（学习率衰减 0.8）
- 符合赛规"至少有 3 次读历史—做实验—依据结果决策的完整记录"要求

### 6. 自进化主循环 (SelfEvolutionHarness)

执行完整的 6 步闭环：
1. 漂移监测
2. 诊断归因
3. 策略选择
4. 实验优化（调用扩展训练器）
5. 退化检查
6. 评估上线（通过则更新检查点，否则回滚）

## 使用方法

### 快速开始

```bash
# B 题自进化（10 轮）
python -m harness.self_evolution \
  --task B \
  --rounds 10 \
  --checkpoint cpu-b-rul-20261110-v1/checkpoint.npz \
  --train cpu-b-rul-20261110-v1/public/train.jsonl \
  --validation cpu-b-rul-20261110-v1/public/validation.jsonl \
  --output submission_runs/B
```

### 参数说明

```
--task {A,B,C}          赛题任务: A(气体识别) / B(轴承RUL) / C(发动机RUL)
--rounds N              自进化轮数 (默认 10)
--checkpoint PATH       基线检查点路径
--train PATH            训练集 JSONL 路径
--validation PATH       验证集 JSONL 路径
--output PATH           输出目录
```

### 输出结构

```
submission_runs/B/
├── experiment_history.jsonl    # 实验历史记录（JSONL 格式）
├── round-01/                   # 第 1 轮输出
│   ├── checkpoint.npz
│   ├── training.json
│   └── history.jsonl
├── round-02/
│   └── ...
└── round-10/
    └── ...
```

## 实验历史记录格式

每轮实验记录包含：
```json
{
  "stage_id": "round-01",
  "strategy": {
    "name": "B_optimal",
    "sample_weight": "none",
    "gauss_aug": 0.0,
    "cosine_anneal": false,
    "shrink_a": 0.45,
    "shrink_center": "median",
    "target_transform": "identity",
    "epochs": 120,
    "learning_rate": 0.01,
    "batch_size": 128
  },
  "validation_rmse": 452.64,
  "validation_macro_f1": null,
  "nasa_score": null,
  "degradation_checks": {
    "iqr_collapse": {
      "collapsed": false,
      "ratio": 0.85,
      "pred_iqr": 420.5,
      "true_iqr": 495.2
    },
    "constant_equivalence": {
      "equivalent_to_constant": false,
      "rmse_model": 452.64,
      "rmse_const": 470.86
    },
    "bias_per_bearing": {
      "Bearing1_4": 166.5,
      "Bearing2_4": 326.3,
      "Bearing3_4": -427.4
    },
    "max_abs_bias": 427.4
  },
  "passed": true,
  "notes": "诊断: life_gap + cross_bearing_level_misalignment"
}
```

## 与 SwanLab 集成

实验记录自动写入 `experiment_history.jsonl`，可通过 SwanLab 的自定义指标接口上传：

```python
import swanlab

swanlab.init(project="industrial-model-evolution")

for record in harness.experiment_manager.history:
    swanlab.log({
        f"{record.stage_id}/rmse": record.validation_rmse,
        f"{record.stage_id}/strategy": record.strategy.name,
        f"{record.stage_id}/passed": record.passed
    })
```

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

## 技术路线论证

### 为什么选择这些策略？

**A 题**：诊断发现 batch2 的 Acetaldehyde 质心离 batch1 的 Ethanol 比离自己类更近（7.50 vs 9.90），oracle LDA 跨批次准确率 0.3663 < 随机 0.6007，证明**决策边界完全不可迁移**。标签移位校正（BBSE/EM）连 ORACLE 都只有 0.5912 < 基线 0.7140，证明是**协变量漂移而非标签移位**。因此采用类平衡加权 + 高斯增强。

**B 题**：诊断发现 Bearing3_4 占验证集 90.2%，且落在训练集 532→2495 分钟的寿命空档中央。rms 与 RUL 秩相关仅 +0.11（另两根轴承 +0.75/+0.96），证明**信号本身缺失**。向训练集中位数收缩可将 RMSE 从 681 降到 452（-33.5%），且**收缩可精确折叠进检查点参数**（误差 ~1e-13），无需改预测代码。

**C 题**：诊断发现 [400,900) 区间仅 44 个样本低估约 190 循环，NASA 惩罚 exp(190/13) ≈ 2.2e6 每样本，主导了整个分数。长寿命加权 + 长时训练可将 NASA 从 5.823e9 降到 3.448e9（-40.8%），RMSE 从 52.87 降到 50.54（-4.4%）。

### 方法论教训（写进技术报告）

1. **单次 train/val 划分会给出与 LOBO 完全相反的结论**  
   → 按整根轴承划分的任务必须用 LOBO

2. **"按验证集挑最优轮次"是隐式早停，数字系统性乐观**  
   → 所有报告数字必须标注 `finalEpVal`（可部署）还是 `bestEpVal`（乐观）

3. **指标变好不等于方法有效，要检查是否退化成常数**  
   → 三道闸门：预测 IQR vs 真值 IQR、分轴承 bias、等价纯常数 RMSE

4. **pooled LOBO 会被巨型轴承污染**  
   → 按寿命分组报告（short/mid/long），并同时看 macro 与 worst-bearing

## 故障排查

### 训练失败

```bash
# 检查 Python 路径
C:\Users\1\anaconda3\python.exe -V  # 必须 3.12.4

# 检查 NumPy 版本
C:\Users\1\anaconda3\python.exe -c "import numpy; print(numpy.__version__)"  # 必须 1.26.4
```

### 余弦退火抛错

```
ValueError: invalid optimizer/scaler state
```

**原因**：学习率衰减到 0 违反 `0 < learning_rate <= 1`  
**解决**：已内置 `max(lr, 1e-6)` 下限

### 检查点不匹配

```
ValueError: checkpoint task, profile, feature schema or classes do not match
```

**原因**：续训时 profile 或 task 不一致  
**解决**：确保 `--task` 与检查点的 `profile` 字段一致

## 版本历史

- **v1.0.0** (2026-11-16): 初始版本，实现完整 6 步闭环
