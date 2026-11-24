# 来源声明与合规清单

**队伍 070228（北京印刷学院）｜版本 HAR-02｜日期 2026-11-24｜对应提交件 `harness-2.0.2.tar`（tag `harness:1.0.0`）**

本文件说明本作品使用的数据、模型、第三方组件与大模型服务来源，并逐条声明赛规要求的数据隔离与合规边界。技术细节见 [`technical_report.md`](technical_report.md)，运行方式见 [`run_manual.md`](run_manual.md)。

---

## 1. 数据集来源与许可

三个数据集均由组委会赛题包下发（`cpu-a-guide-20261110-v1`、`cpu-b-rul-20261110-v1`、`cpu-c-guide-20261110-v1`），包内只有 `public/train.jsonl`、`public/validation.jsonl`、`checkpoint.npz`、`training.json`、`history.jsonl` 与基线代码。本队**未修改赛题包任何文件**。

| 数据集 | 公开来源 | 原始形态 | 引用与使用边界 |
|---|---|---|---|
| A · 气体识别 | UCI Machine Learning Repository **#224** *Gas Sensor Array Drift at Different Concentrations*（128 维 = 16 传感器 × 8 特征，6 类气体） | 原始数据文件 + 批次标注 | 按 UCI 仓库页面声明（CC BY 4.0）引用；本队只使用平台下发的 JSONL 投影，未下载、未再分发原始数据 |
| B · 轴承 RUL | **XJTU-SY** 滚动轴承加速寿命试验数据集（西安交通大学，王彪、雷亚国等公开） | 原始振动 CSV（按分钟快照） | 引用原始论文（Wang B., Lei Y., Li N., Li N., *A hybrid prognostics approach for estimating remaining useful life of rolling element bearings*, IEEE Trans. Reliability, 2020）；公开供研究使用，本队未读取原始 CSV、未再分发 |
| C · 发动机 RUL | **NASA C-MAPSS** 涡扇发动机退化仿真数据（FD001–FD004，NASA Prognostics Center of Excellence） | 原始 train/test/RUL txt | 引用 Saxena A., Goebel K., Simon D., Eklund N., *Damage propagation modeling for aircraft engine run-to-failure simulation*, PHM 2008；本队只使用平台 JSONL 投影 |

**说明**：为避免任何"用外部数据补隐藏测试"的嫌疑，本队**从未打开**桌面上存放的 XJTU-SY / C-MAPSS 原始 CSV 目录；所有训练、诊断、选型只发生在赛题包 `public/` 投影上。任务包内出现的轴承只有公开的 12 根（`Bearing1_1`…`3_4`），平台未公开的隐藏轴承从未出现在任何本地实验中。

---

## 2. 模型来源与改动

基础模型全部是**组委会任务包自带**的 CPU NumPy MLP（`baselines/model.py`，`contest.cpu-mlp.v1` 格式、Adam、`npz` 无 pickle），父检查点为包内 `checkpoint.npz`。

| 任务 | 结构 | 参数量 | 本版改动 | 合规性 |
|---|---|---|---|---|
| A | 128 → 32 tanh → 6 softmax | 4,326 | 不改结构；只改训练过程（invfreq 类平衡、高斯噪声、通道 dropout、快照选型） | ✅ 续训合同允许 |
| B | 34 → 24 tanh → 1 linear | 865 | 不改结构；目标改为寿命分数 `c·y/D_b`，**只折叠进 `target_scale/target_mean`**，`model.py`/`predict.py` 一行未改 | ✅ 检查点格式兼容，预测仍以分钟输出 |
| C | 24 → 32 tanh → 1 linear | 833 | 不改结构；只改训练过程（全量、batch 256、cosine、逐 epoch 选轮） | ✅ 续训合同允许 |

- 三题候选都是 `Model.load(父)` 后**继续训练**（保留 Adam 一/二阶矩与 `mean/scale`），不是重新初始化；检查点 `epochs`/`steps` 相对父模型单调增。
- A 题的 hidden 32→128 拓宽是"显式兼容的权重迁移"（新列/新行按函数保持方式填充并更新 `parameter_count`），但实测无稳定增益，**默认关闭**；B/C 的拓宽路线经实测否决，未启用。
- B 题的 `D_b` 由**训练标签**推出（组内 RUL=0 行→`max(y)+1`；片段→`y+快照号`）；快照号只用于还原标签尺度，**从不作为模型输入**（题面禁止）。C 题的 `meta.source_subset` 只用于报告分组，从不进入特征。

---

## 3. 第三方组件与许可

镜像基础层为 Python 3.12.4 官方镜像（Debian 12.6），其自带组件的许可证文件随镜像保留。主流程（`/app` 内代码）**只依赖 numpy**。

| 组件 | 镜像内版本 | 许可证 | 用途 | 合规 |
|---|---|---|---|---|
| Python | 3.12.4 | PSF License | 运行时 | ✅ |
| NumPy | 1.26.4 | BSD 3-Clause | 全部数值计算（唯一主流程依赖） | ✅ 保留版权与许可声明 |
| SciPy | 1.17.1 | BSD 3-Clause | 基础层随附，主流程未调用 | ✅ |
| scikit-learn | 1.9.1 | BSD 3-Clause | 基础层随附，主流程未调用 | ✅ |
| swanlab | 0.10.5 | Apache-2.0 | 基础层随附；本版主流程未上传任何记录 | ✅ |
| 其余基础层包（requests/urllib3/pydantic 等） | 见镜像 | 各自 MIT/BSD/Apache | 基础层依赖，未用于训练或预测 | ✅ 保留原许可文件 |

主流程依赖（numpy）与基础层主要组件均为 BSD/MIT/Apache/PSF 类许可，未主动引入 GPL/AGPL/非商业许可组件；未使用 PyTorch、TensorFlow、HuggingFace 预训练权重或任何外部模型权重。

---

## 4. SDK 与平台接口

| 项 | 值 |
|---|---|
| SDK | `contest_harness` **1.0.2**（镜像内 site-packages；`describe` 输出 `sdk_version:"1.0.2"`） |
| 协议 | `contest.harness.workspace.v2`；prompt_version `industrial-workspace-flexible-v3` |
| 入口 | `/app/harness.py describe\|train\|predict`；选手模块 `/app/contest_submission.py` |
| 上传协议 | 1.0.2 内联（`candidate_request`/`state_request` 自带 `files`）；**未注入 1.0.4**，避免宿主未宣告 `chunked-v1` 时抛 `CHUNKED_UPLOAD_UNSUPPORTED` |
| 平台校验 | 正式评测 30 轮全通（宿主接受 1.0.2 内联），本地仿真 30/30 复现 |

---

## 5. 大模型来源与使用边界

| 项 | 内容 |
|---|---|
| 模型 | **Qwen/Qwen3.5-4B**（平台白名单内的基础模型，`rnd.model_name` 由宿主下发） |
| 调用方式 | 每轮至少 1 次 `rnd.model(messages, max_tokens, temperature=0)`；`finish.model_call_ids` 引用本轮真实回执号 |
| 输入 | 本轮数据集代号、阶段、父/候选指标摘要、实验元数据、历史轮次摘要（**不含训练数据原文、不含检查点权重、不含密钥**） |
| 输出 | 一段决策说明（"为什么提交/保留"），写入 `finish.reason` 与 `hypothesis/change_summary`；**不参与前向预测、不写入检查点、不影响 `predict` 输出** |
| 密钥 | PyTRiO Key 只由平台提交页保管；镜像、源码、文档、训练记录中均不含任何密钥 |

---

## 6. 合规声明（逐条）

| # | 声明 | 事实依据 |
|---|---|---|
| 1 | 未使用、未推断隐藏测试标签 | 赛题包内只挂载 `public/` 两个文件；隐藏集不进训练容器；本队未按文件名/样本号反推 |
| 2 | 未读取桌面原始 CSV | 全部实验脚本只读 `work/a`、`work/b`、`work/c` 与赛题包 `public/`；XJTU-SY / C-MAPSS 原始目录从未被打开 |
| 3 | 公开验证集不并入梯度训练 | 三个 `strategy.py` 只在 `train_rows`/`x_train` 上求梯度；val 只用于指标、选 epoch 与门槛 |
| 4 | 父检查点真实续训 | `Model.load(父)` 后继续；保留 Adam 矩与归一化统计量；候选与父字节不同（A 已验收） |
| 5 | 未重拟合归一化统计量 | A 实测重拟合会掉到 0.5221，因此三题一律沿用父检查点 `mean/scale` |
| 6 | 不改特征 schema / 任务类型 | 特征只来自 `inputs`，顺序按检查点 `meta.paths`；`task` 以 `rnd.task` 为准 |
| 7 | 不联网 | 镜像内不发起外部请求；仿真与本机实验全程离线 |
| 8 | 不写 `/app`、不改 `/previous` | 候选/状态只在 `/workspace`（或 `tempfile` 目录）生成；`Model.save` 目标必须不存在 |
| 9 | 不把密钥写入镜像/源码/文档 | 见 §5 |
| 10 | 每轮至少 1 次真实大模型调用 | 快进轮也先调用再 `finish`；30/30 轮 `model_call_ids` 非空且为本轮真实回执 |
| 11 | 实验记录可复算 | `evidence/report-{a,b,c}.md`、`evidence/sim-app202-summary.md`、`evidence/round-history/{A,B,C}.jsonl` 随源码仓库提交，含逐轮帧摘要、逐 epoch 轨迹与 495 子集缓存汇总 |
| 12 | 结果口径诚实 | 验证最优轮与固定末轮两个口径都报；val 选型偏差、小类不确定性、短轴承代价均在报告中披露 |

---

## 7. 原创性与引用

| 组件 | 原创性 | 说明 |
|---|---|---|
| `contest_submission.py`（接入层、投影 digest、快进、预算、门槛调度） | 独立编写 | 本版新增的父模型保留与快进机制 |
| `strategies/{a,b,c}_impl.py`、`core.py` | 独立编写 | 三题训练器、目标重定义、配对 bootstrap、IQR/NASA 守卫 |
| `baselines/` | 基于赛题包官方基线 | 数据适配与 `Model` 语义未改 |
| 本套文档 | 独立编写 | — |

引用：组委会赛题包与基线代码；UCI #224；XJTU-SY（Wang et al., 2020）；NASA C-MAPSS（Saxena et al., 2008）；平台 PyTRiO / Qwen3.5-4B 服务按其条款使用。

---

## 8. 联系

如有合规疑问，请通过大赛提交平台联系队伍 070228。本声明如与平台最终认定不一致，以平台认定为准。
