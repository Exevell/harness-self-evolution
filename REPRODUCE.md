# 复现说明

## 环境要求

### 硬件
- **CPU**: 任意 x86_64（基线仅需 CPU）
- **内存**: ≥ 4GB（C 题训练集 113k 样本）
- **磁盘**: ≥ 2GB（三个赛题包 + 输出）

### 软件
- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.12.4（**必须**，参考环境锁定版本）
- **NumPy**: 1.26.4（**必须**，参考环境锁定版本）
- **Scikit-learn**: ≥ 1.3.0（用于 LDA 诊断）

### ⚠️ 关键约束
- **Python 路径**: 必须使用 `C:\Users\1\anaconda3\python.exe`
  - 系统 `python` 命令指向 `WindowsApps\python.exe` 存根，会**静默失败（exit 1）**
- **NumPy 版本**: 必须 1.26.4，其他版本可能导致随机数不一致
- **学习率下限**: 余弦退火时 `max(lr, 1e-6)`，否则 `model.validate()` 抛 `invalid optimizer/scaler state`

## 安装依赖

```bash
# 1. 安装基础依赖
C:\Users\1\anaconda3\python.exe -m pip install -r requirements.txt

# 2. 验证环境
C:\Users\1\anaconda3\python.exe -V                    # Python 3.12.4
C:\Users\1\anaconda3\python.exe -c "import numpy; print(numpy.__version__)"  # 1.26.4
```

## 快速复现

### 1. B 题（轴承 RUL）- 推荐首先复现

```bash
# 使用一键运行脚本
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task B \
  --rounds 10 \
  --base-dir C:\Users\1\Downloads \
  --output-dir submission_runs/B
```

**预期输出**:
```
自进化 Harness - 一键运行脚本
======================================================================
任务: B (轴承 RUL)
轮数: 10
赛题包: C:\Users\1\Downloads\cpu-b-rul-20261110-v1
检查点: C:\Users\1\Downloads\cpu-b-rul-20261110-v1\checkpoint.npz
训练集: C:\Users\1\Downloads\cpu-b-rul-20261110-v1\public\train.jsonl
验证集: C:\Users\1\Downloads\cpu-b-rul-20261110-v1\public\validation.jsonl
输出目录: submission_runs/B
评估指标: rmse
======================================================================

============================================================
自进化轮次: round-01
============================================================

[1/6] 漂移监测...
  最大 MSE 占比: 44.5% (阈值 80.0%) → 无主导轴承
[2/6] 诊断归因...
  根因: unknown (none)
[3/6] 策略选择...
  调用 PyTRIO 大模型（Qwen/Qwen3.5-4B @ https://pytrio.com/api/openai/v1）...
  ✅ 大模型决策: B_shrink030 {'shrink_a': 0.3} (1.2s)
     理由: 诊断显示无显著主导轴承且当前策略为B_shrink045，降低shrink_a至0.3可增强模型对非主导样本的拟合能力以改善RMSE
[4/6] 实验优化...
[5/6] 退化检查...
  ⚠️ 预测分布坍缩 (IQR 67.18 / 真值 839, ratio 0.080)
[6/6] 评估上线...
  ❌ 上线失败，回滚到上一轮检查点

📊 轮次 1/3 结果:
   RMSE: 458.48
   通过: ❌
   策略: B_shrink030

⚠️  轮次 1 未通过退化检查，已回滚，继续下一轮探索

...
======================================================================
自进化完成
======================================================================
总轮数: 3
通过轮数: 0

实验记录: submission_runs/B/experiment_history.jsonl
输出目录: submission_runs/B
```

**实测时间**: B 题三轮合计约 1 分钟（纯 CPU，样本 7032 / 1679）。**实测结果**: 458.48 / 467.64 / 486.06，三轮全部回滚（原因见 `degradation_checks`）。

### 2. A 题（气体识别）

```bash
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task A \
  --rounds 1 \
  --base-dir C:\Users\1\Downloads \
  --output-dir submission_runs/A
```

**实测结果**: Macro-F1 **0.7412**（官方基线 0.7140，+3.8%），1 轮通过闸门并上线。
**实测时间**: 单轮 1 分钟左右。

### 3. C 题（发动机 RUL）

```bash
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task C \
  --rounds 3 \
  --base-dir C:\Users\1\Downloads \
  --output-dir submission_runs/C
```

**实测结果**: RMSE 92.86 / 262.45 / 101.12，全部劣于可部署常数 79.41，三轮全部回滚。
**实测时间**: 三轮约 3～5 分钟。
**说明**: C 题策略池当前与基线时间尺度不匹配（基线只训 15 epoch，策略一律 300 epoch），这是已知局限，不做美化。

## 手动复现（逐步）

如果需要更细粒度的控制，可以手动调用扩展训练器：

### B 题手动复现

```bash
# 第 1 轮
C:\Users\1\anaconda3\python.exe -m baselines.train \
  --profile B \
  --train C:\Users\1\Downloads\cpu-b-rul-20261110-v1\public\train.jsonl \
  --validation C:\Users\1\Downloads\cpu-b-rul-20261110-v1\public\validation.jsonl \
  --from-baseline C:\Users\1\Downloads\cpu-b-rul-20261110-v1\checkpoint.npz \
  --epochs 120 \
  --learning-rate 0.01 \
  --batch-size 128 \
  --shrink-a 0.45 \
  --shrink-center median \
  --output submission_runs/B/round-01 \
  --stage-id round-01

# 第 2 轮（续训）
C:\Users\1\anaconda3\python.exe -m baselines.train \
  --profile B \
  --train C:\Users\1\Downloads\cpu-b-rul-20261110-v1\public\train.jsonl \
  --validation C:\Users\1\Downloads\cpu-b-rul-20261110-v1\public\validation.jsonl \
  --resume submission_runs/B/round-01/checkpoint.npz \
  --epochs 120 \
  --learning-rate 0.008 \
  --batch-size 128 \
  --shrink-a 0.45 \
  --shrink-center median \
  --output submission_runs/B/round-02 \
  --stage-id round-02
```

### A 题手动复现

```bash
C:\Users\1\anaconda3\python.exe -m baselines.train \
  --profile A \
  --train C:\Users\1\Downloads\cpu-a-guide-20261110-v1\public\train.jsonl \
  --validation C:\Users\1\Downloads\cpu-a-guide-20261110-v1\public\validation.jsonl \
  --from-baseline C:\Users\1\Downloads\cpu-a-guide-20261110-v1\checkpoint.npz \
  --epochs 250 \
  --learning-rate 0.003 \
  --batch-size 64 \
  --sample-weight invfreq \
  --gauss-aug 0.5 \
  --cosine-anneal \
  --output submission_runs/A/round-01 \
  --stage-id round-01
```

### C 题手动复现

```bash
C:\Users\1\anaconda3\python.exe -m baselines.train \
  --profile C \
  --train C:\Users\1\Downloads\cpu-c-guide-20261110-v1\public\train.jsonl \
  --validation C:\Users\1\Downloads\cpu-c-guide-20261110-v1\public\validation.jsonl \
  --from-baseline C:\Users\1\Downloads\cpu-c-guide-20261110-v1\checkpoint.npz \
  --epochs 300 \
  --learning-rate 0.003 \
  --batch-size 128 \
  --sample-weight rul-150x3 \
  --output submission_runs/C/round-01 \
  --stage-id round-01
```

## 验证结果

### 检查输出目录结构

```bash
dir submission_runs\B
```

预期:
```
round-01/
round-02/
...
round-10/
experiment_history.jsonl
```

### 检查单轮输出

```bash
dir submission_runs\B\round-01
```

预期:
```
checkpoint.npz       # 检查点
training.json        # 训练配置 + 验证指标
history.jsonl        # 逐 epoch 记录
```

### 检查实验历史

```bash
type submission_runs\B\experiment_history.jsonl
```

预期（每行一条 JSON 记录，字段名为真实结构）:
```json
{"stage_id": "round-01", "strategy": {"name": "B_shrink030", "shrink_a": 0.3, ...}, "validation_metric": 458.47844336937425, "passed": false, "degradation_checks": {"iqr_collapse": {"collapsed": true, "ratio": 0.08007084509844}, ...}, "llm": {"source": "llm", "latency_s": 1.2}, ...}
{"stage_id": "round-02", "strategy": {"name": "B_shrink045", ...}, "validation_metric": 467.63664511562735, "passed": false, ...}
{"stage_id": "round-03", "strategy": {"name": "B_shrink060", ...}, "validation_metric": 486.0615916351393, "passed": false, ...}
```

> 注意：暂无可上线轮次时，`passed` 全为 `false`，这是闸门在起作用，不是运行失败。

### 检查 training.json

```bash
type submission_runs\B\round-01\training.json
```

预期包含:
- `parent_checkpoint_sha256`: 父检查点哈希
- `checkpoint_sha256`: 当前检查点哈希
- `validation_before`: 训练前验证指标
- `validation_after`: 训练后验证指标
- `extensions`: 扩展配置（sample_weight, shrink_a 等）

## 常见问题

### Q: 训练失败 "invalid optimizer/scaler state"

**原因**: 学习率衰减到 0  
**解决**: 已内置 `max(lr, 1e-6)` 下限，检查代码是否正确

### Q: 检查点不匹配 "checkpoint task, profile, feature schema or classes do not match"

**原因**: 续训时 profile 或 task 不一致  
**解决**: 确保 `--profile` 与检查点的 `profile` 字段一致

### Q: 输出目录已存在

**原因**: `train.py:43` 检查 `output.exists()` 并拒绝覆盖  
**解决**: 删除旧目录或使用新目录名

### Q: 复现结果与报告略有差异

**原因**: 随机数生成器差异  
**解决**: 确保使用 Python 3.12.4 + NumPy 1.26.4，所有超参数与报告一致

### Q: 如何从断点恢复？

```bash
# 使用 --resume 参数
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task B \
  --rounds 10 \
  --resume
```

## 性能基准

| 任务 | 官方基线 | 本系统实测 | 相对基线 | 闸门结论 |
|------|----------|------------|----------|----------|
| A（气体识别） | 0.7140 (Macro-F1) | **0.7412** | +3.8% | ✅ 1/1 上线 |
| B（轴承 RUL） | 681.16 (RMSE) | 458.48 / 467.64 / 486.06 | −32.7% / −31.4% / −28.6% | ❌ 0/3 回滚 |
| C（发动机 RUL） | 52.87 (RMSE) | 92.86 / 262.45 / 101.12 | 未达标 | ❌ 0/3 回滚 |

数字来源：官方基线取任务包 `training.json` 的 `validation_after`；实测取 `submission_runs/*/experiment_history.jsonl`。旧版本的 0.7519 / 452.64 / 50.54 / NASA 3.448e9 无法在当前镜像复现，已撤回。

**测试环境**:
- CPU: Intel Core i7-12700H
- 内存: 16GB
- 操作系统: Windows 11
- Python: 3.12.4
- NumPy: 1.26.4

## 技术支持

- **赛题文档**: `cpu-*/TASK.md`, `cpu-*/TRAINING_GUIDE.md`
- **Harness 文档**: `harness/README.md`
- **分析报告**: `progress-report-工业模型自进化赛题.md`

## 版本信息

- **Harness 版本**: 1.0.0
- **提交日期**: 2026-11-16
- **Git 提交**: （待填写）
