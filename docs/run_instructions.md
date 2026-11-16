# 运行说明

## 1. 环境准备

### 1.1 硬件要求

| 资源 | 最低要求 | 推荐配置 |
|------|---------|---------|
| CPU | x86_64 任意 | Intel Core i7-12700H |
| 内存 | 4GB | 16GB |
| 磁盘 | 2GB | 10GB |

### 1.2 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | **3.12.4** | 必须，参考环境锁定版本 |
| NumPy | **1.26.4** | 必须，参考环境锁定版本 |
| scikit-learn | ≥1.3.0 | 用于 LDA 诊断 |
| swanlab | ≥0.3.0 | 实验记录上传 |

### 1.3 安装步骤

```bash
# 1. 验证 Python 版本（必须使用 anaconda 路径）
C:\Users\1\anaconda3\python.exe -V
# 期望输出: Python 3.12.4

# 2. 验证 NumPy 版本
C:\Users\1\anaconda3\python.exe -c "import numpy; print(numpy.__version__)"
# 期望输出: 1.26.4

# 3. 安装依赖
C:\Users\1\anaconda3\python.exe -m pip install -r requirements.txt
```

### 1.4 ⚠️ 关键约束

- **Python 路径**：必须使用 `C:\Users\1\anaconda3\python.exe`
  - 系统 `python` 命令指向 `WindowsApps\python.exe` 存根，会**静默失败（exit 1）**
- **NumPy 版本**：必须 1.26.4，其他版本可能导致随机数不一致
- **学习率下限**：余弦退火时 `max(lr, 1e-6)`，否则 `model.validate()` 抛错

---

## 2. 快速运行

### 2.1 一键运行脚本

```bash
# B 题自进化（10 轮）
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task B \
  --rounds 10 \
  --base-dir C:\Users\1\Downloads \
  --output-dir submission_runs/B
```

**参数说明**：
- `--task {A,B,C}`：赛题任务
- `--rounds N`：自进化轮数（默认 10）
- `--base-dir PATH`：赛题包所在基础目录（默认 `C:\Users\1\Downloads`）
- `--output-dir PATH`：输出目录（默认 `submission_runs/{task}`）
- `--resume`：从已有实验记录恢复

### 2.2 预期输出

```
======================================================================
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
[2/6] 诊断归因...
  根因: life_gap + cross_bearing_level_misalignment
  证据: 主导轴承 Bearing3_4 占 MSE 90.2%，且落在寿命空档 (532, 2495, 1963)
  动作: shrink_toward_median
[3/6] 策略选择...
  策略: B_optimal
[4/6] 实验优化...
[5/6] 退化检查...
[6/6] 评估上线...
  ✅ 上线成功 (RMSE=452.64)

📊 轮次 1/10 结果:
   RMSE: 452.64
   通过: ✅
   策略: B_optimal

...

======================================================================
自进化完成
======================================================================
总轮数: 10
通过轮数: 10
最优 RMSE: 452.64
最优策略: B_optimal
最优轮次: round-01

实验记录: submission_runs/B/experiment_history.jsonl
输出目录: submission_runs/B

======================================================================
下一步: 上传到 SwanLab
======================================================================
python upload_to_swanlab.py --history submission_runs/B/experiment_history.jsonl
```

### 2.3 预期时间

| 任务 | 单轮耗时 | 10 轮总耗时 |
|------|---------|------------|
| A（气体识别） | ~12 秒 | ~2 分钟 |
| B（轴承 RUL） | ~30 秒 | ~5 分钟 |
| C（发动机 RUL） | ~90 秒 | ~15 分钟 |

---

## 3. 手动运行（逐步）

如果需要更细粒度的控制，可以手动调用扩展训练器：

### 3.1 B 题手动复现

```bash
# 第 1 轮
C:\Users\1\anaconda3\python.exe -m baselines.train_ext \
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
C:\Users\1\anaconda3\python.exe -m baselines.train_ext \
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

### 3.2 A 题手动复现

```bash
C:\Users\1\anaconda3\python.exe -m baselines.train_ext \
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

### 3.3 C 题手动复现

```bash
C:\Users\1\anaconda3\python.exe -m baselines.train_ext \
  --profile C \
  --train C:\Users\1\Downloads\cpu-c-guide-20261110-v1\public\train.jsonl \
  --validation C:\Users\1\Downloads\cpu-c-guide-20261110-v1\public\validation.jsonl \
  --from-baseline C:\Users\1\Downloads\cpu-c-guide-20261110-v1\checkpoint.npz \
  --epochs 300 \
  --learning-rate 0.003 \
  --batch-size 128 \
  --sample-weight rul-150x3 \
  --cosine-anneal \
  --output submission_runs/C/round-01 \
  --stage-id round-01
```

---

## 4. 验证结果

### 4.1 检查输出目录结构

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

### 4.2 检查单轮输出

```bash
dir submission_runs\B\round-01
```

预期:
```
checkpoint.npz       # 检查点
training.json        # 训练配置 + 验证指标
history.jsonl        # 逐 epoch 记录
```

### 4.3 检查实验历史

```bash
type submission_runs\B\experiment_history.jsonl
```

预期（每行一条 JSON 记录）:
```json
{"stage_id": "round-01", "strategy": {"name": "B_optimal", ...}, "validation_rmse": 452.64, "passed": true, ...}
{"stage_id": "round-02", "strategy": {"name": "B_optimal_tuned", ...}, "validation_rmse": 452.80, "passed": true, ...}
...
```

### 4.4 检查 training.json

```bash
type submission_runs\B\round-01\training.json
```

预期包含:
- `parent_checkpoint_sha256`: 父检查点哈希
- `checkpoint_sha256`: 当前检查点哈希
- `validation_before`: 训练前验证指标
- `validation_after`: 训练后验证指标
- `extensions`: 扩展配置（sample_weight, shrink_a 等）

---

## 5. 上传到 SwanLab

### 5.1 安装 SwanLab

```bash
C:\Users\1\anaconda3\python.exe -m pip install swanlab
```

### 5.2 上传实验历史

```bash
C:\Users\1\anaconda3\python.exe upload_to_swanlab.py \
  --history submission_runs/B/experiment_history.jsonl \
  --project industrial-model-evolution
```

**预期输出**:
```
📂 读取到 10 条实验记录
🎯 任务: B
📊 项目: industrial-model-evolution

  ✅ 上传 round-01
  ✅ 上传 round-02
  ...
  ✅ 上传 round-10

📈 计算最优策略...
  🏆 最优 RMSE: 452.64 (round-01)

✅ 上传完成!
📊 查看结果: https://swanlab.cn/@your-username/industrial-model-evolution
```

---

## 6. 故障排查

### 6.1 训练失败 "invalid optimizer/scaler state"

**原因**：学习率衰减到 0  
**解决**：已内置 `max(lr, 1e-6)` 下限，检查代码是否正确

### 6.2 检查点不匹配 "checkpoint task, profile, feature schema or classes do not match"

**原因**：续训时 profile 或 task 不一致  
**解决**：确保 `--profile` 与检查点的 `profile` 字段一致

### 6.3 输出目录已存在

**原因**：`train.py:43` 检查 `output.exists()` 并拒绝覆盖  
**解决**：删除旧目录或使用新目录名

### 6.4 复现结果与报告略有差异

**原因**：随机数生成器差异  
**解决**：确保使用 Python 3.12.4 + NumPy 1.26.4，所有超参数与报告一致

### 6.5 如何从断点恢复？

```bash
C:\Users\1\anaconda3\python.exe run_harness.py \
  --task B \
  --rounds 10 \
  --resume
```

---

## 7. 性能基准

| 任务 | 基线 | 最优 | 提升 | 单轮耗时 |
|------|------|------|------|---------|
| A（气体识别） | Macro-F1 0.7140 | **0.7519** | +5.3% | ~12 秒 |
| B（轴承 RUL） | RMSE 681.16 | **452.64** | -33.5% | ~30 秒 |
| C（发动机 RUL） | RMSE 52.87 / NASA 5.823e9 | **50.54 / 3.448e9** | -4.4% / -40.8% | ~90 秒 |

**测试环境**：
- CPU: Intel Core i7-12700H
- 内存: 16GB
- 操作系统: Windows 11
- Python: 3.12.4
- NumPy: 1.26.4

---

## 8. 联系与支持

- **赛题文档**：`cpu-*/TASK.md`, `cpu-*/TRAINING_GUIDE.md`
- **Harness 文档**：`harness/README.md`
- **技术报告**：`docs/technical_report.md`
- **设计论证**：`docs/design_justification.md`
