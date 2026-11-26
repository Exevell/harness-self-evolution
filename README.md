# HAR-02 · 工业模型自进化 Harness（队伍 070228）

面向「智训引擎」工业模型自进化赛题的参赛系统。平台在**漂移数据流**上对每个数据集跑 10 轮：
每轮宿主下发当前阶段的公开 `train/validation` 投影、父检查点和历史记录，容器内执行
`python3 /app/harness.py train`，选手从**平台提供的父检查点续训**并选择本轮产物。

本仓库是 **HAR-02** 版本的完整源码：接入层、三题策略、基线与复现材料。

## 1. 这一版修掉了什么

上一版（HAR-01）正式评测客观分为 12.4576/35，平台判定 `not_below_static_baseline=false`。
根因有三条，全部在代码层可查：

1. **每轮只训 1–4 个 epoch**（`_plans()` 网格），10 轮累计约 20 epoch，父模型本身已训 15–20 epoch；
2. **C 题截断**：`TRAIN_CAP=12000` 把 113,089 行训练集截成按文件顺序的前 12,000 行，
   `VAL_CAP=3000` 把 23,049 行验证集截成前 3,000 行 —— 有序子集分布偏斜，模型坍缩成常数；
   平台报告显示那段 3k val 上父模型 RMSE 仅 43.09，而我们十轮提交的是 55–58，**每轮都更差**；
3. **没有回滚**：`best` 只在"训练过的候选"里选，兜底强制训练 1 epoch 后提交，
   于是出现候选比父模型更差还上线（B 题 r5/r8/r10）。

HAR-02 的对应设计：

| 机制 | 实现 | 目的 |
|---|---|---|
| 父模型保留 | 每轮先在同一 val 上评测父模型；候选超过边际才提交，否则 `selected_artifact_id=rnd.parent.artifact_id` | 把单轮下降风险封顶为「不提交」 |
| 全量数据 | 取消一切行数截断 | 消除有序子集偏置 |
| 相同投影快进 | 对投影算 digest，命中且上一轮已判保留 → 本轮只做 1 次模型调用后保留父产物 | 资源效率 / 避免无信息重复搜索 |
| 预算感知 | 按 `rnd.wall_seconds` 分配训练预算（≤40%，上限 180s），按 chunk 计时早停 | 2 vCPU 正式环境不超时 |
| 三题专用策略 | `strategies/{a,b,c}_impl.py` | 见下 |

## 2. 三题方法

### A · 气体识别（macro-F1）

父模型在 batch2 上 **Acetaldehyde 召回为 0**（109 例中 108 例判成 Ethanol）——这是类别级协变量漂移，
无监督对齐救不回来（CORAL 0.31，批均值对齐 0.62，oracle 类别偏置上界 +0.0136）。
有效手段是正则化适配：逆频率类平衡 + 高斯噪声 σ=0.8 + 通道 dropout 0.1，约 1200 梯度步，
再以**配对 bootstrap（P>0.8）+ 绝对边际 0.008** 决定是否上线。

public validation macro-F1：**0.714030 → 0.755890**（+0.0419）；样本外半分割 +0.0273±0.0195。
重拟合 `mean/scale` 会掉到 0.5221，因此**沿用父检查点归一化**（也符合续训合同）。

### B · 轴承 RUL（RMSE 分钟）

`RUL = D_b − i`，**绝对分钟在轴承间不可辨识**：LOBO 显示预测水平与轴承总寿命 `D_b` 无关。
因此回归**寿命分数** `p = y / D_b`，再用单一先验 `c` 还原分钟：`ŷ = c·p̂`。
它等价于目标仿射变换 `c·y/D_b`，可完全折叠进检查点的 `target_scale/target_mean`，
**不改 `model.py` / `predict.py`**，预测仍以分钟输出。

public validation RMSE：**681.160 → 450.316（−33.9%）**，MAE 563.106 → 337.5；
LOBO（495 个随机 4 根子集）947 → 654（−31%）；上线守卫：相对改善 ≥5% 且预测 IQR/真值 IQR ≥0.35。

### C · 发动机 RUL（RMSE 循环 + NASA 分数）

父模型最大预测仅 298（真值到 434），RUL>300 的 2.9% 样本贡献 94% 的 NASA 分。
全量续训（113,089 行）、batch 256、cosine 3e-3→1e-6、600 epoch、按验证集选最优轮：

public validation：RMSE **52.869 → 50.258（−4.94%）**，MAE **38.900 → 36.010（−7.43%）**，
NASA **5.823e9 → 2.709e9（−53.5%）**；FD001–FD004 四个子集 RMSE 全部改善。
上线守卫：RMSE 改善 ≥1% 且 NASA 不回归。

## 3. 目录

```
contest_submission.py      平台接入层：train(rnd) / predict(sample, model_dir)
harness.py                 固定入口（describe | train | predict），调 contest_harness.cli
run.sh                     镜像 ENTRYPOINT：带参 exec 透传，无参常驻
baselines/                 官方基线：NumPy MLP + Adam + JSONL 数据适配（与赛题包同源）
strategies/__init__.py     候选驱动：plan 表、按数据集分发到专用实现、父模型保留判定
strategies/core.py         fit_ext（样本权重 / 高斯增强 / 余弦退火 / huber）与指标
strategies/a_impl.py       A 题策略（配对 bootstrap 门槛）
strategies/b_impl.py       B 题策略（寿命分数目标 + IQR 守卫）
strategies/c_impl.py       C 题策略（NASA 守卫 + 分块计时早停）
```

## 4. 运行

平台侧（镜像内）：

```bash
python3 /app/harness.py describe    # {"protocol":"contest.harness.workspace.v2", ...}
python3 /app/harness.py train       # 读 /data /previous /state，写 /workspace
python3 /app/harness.py predict     # stdin 逐行 {"task","inputs"} → stdout 逐行 JSON
```

本地复现：把赛题包（`cpu-a|b|c-guide-20261110-v1`）的 `public/` 与 `checkpoint.npz`
挂成 `/data`、`/previous`，或用 `work/harness/platform_sim.py`（开发期工具，不随镜像提交）。

## 5. 合规

- 只使用官方赛题包内的 `public/train.jsonl` 与 `public/validation.jsonl`；
  验证集标签仅用于测量与选型，**不进入梯度训练**；
- 父检查点真实续训（保留优化器状态与归一化统计），不改特征 schema / 任务类型；
- 不读取任何原始 CSV / 隐藏轴承 / 隐藏测试标签；
- 检查点内不写入任何密钥；PyTRiO Key 只由平台提交页保管。

## 6. 镜像

`harness-2.0.2.tar`（`docker save`，linux/amd64，单层扁平，391,372,800 B，
sha256 `56456163261a77242b0a3c1325d0938b8510e6888beacebc034767c2204e8240`），
内含 SDK `contest_harness 1.0.5`，`/app` 12 个文件，无 `__pycache__`、无 SDK 副本。
