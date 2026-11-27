# 系统运行说明（Run Manual）

**队伍 070228（北京印刷学院）｜版本 HAR-04（镜像 `harness-4.2.0.tar`）｜日期 2026-11-27**

本文件说明提交镜像在平台侧如何运行、如何构建与校验镜像、如何在本地复现本报告的全部数字，以及已知限制与故障排查。技术结论见 [`technical_report.md`](technical_report.md)，来源与合规见 [`source_declaration.md`](source_declaration.md)。

---

## 1. 提交件清单

| 提交件 | 位置 | 说明 |
|---|---|---|
| 系统镜像 | **`harness-4.2.0.tar`（391,475,200 B / 373.34 MiB）** | `docker save` 未压缩、linux/amd64、单层扁平；tag `harness:1.0.0`；sha256 `e214ef6a0f282524a60cdb468e2b0ff25ed1ebe891b46978c6b1a62ea979ae75` |
| 源码 | 镜像 `/app` + 源码仓库 | 接入层、三题策略、基线、`stage_policy.py`、`run.sh` |
| 文档 | `docs/technical_report.md`、`docs/source_declaration.md`、`docs/run_manual.md` | 本套材料 |
| 实验结果 | 见技术报告 §6.6 | 逐轮帧、逐 epoch 轨迹、样本外预测缓存 |

镜像内 `/app` 共 **13 个文件（160.4 KiB）**：`harness.py`、`contest_submission.py`、`run.sh`（权限 0755）、`stage_policy.py`、`baselines/`（`__init__`、`data`、`model`、`predict`）、`strategies/`（`__init__`、`a_impl`、`b_impl`、`c_impl`、`core`）。

---

## 2. 平台侧运行

### 2.1 固定入口

```text
python3 /app/harness.py describe    # 打印一行 JSON，声明协议/任务/模型访问方式
python3 /app/harness.py train       # 读 /data /previous /state，把候选/状态写 /workspace
python3 /app/harness.py predict     # stdin 逐行 {"task","inputs"} → stdout 逐行 JSON
```

`describe` 的实际输出：

```json
{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc",
 "sdk":"contest_harness","sdk_version":"1.0.5",
 "tasks":["classification","regression"]}
```

### 2.2 目录与权限（只读 vs 可写）

| 路径 | 权限 | 内容 | 本系统的用法 |
|---|---|---|---|
| `/app` | 只读 | 选手代码、SDK、`harness.py` | 只 import，不写 |
| `/data/train.jsonl`、`/data/validation.jsonl` | 只读 | 当前阶段公开投影 | train 求梯度；validation 只测量/选型 |
| `/previous` | 只读 | 当前父产物（路径与清单一致） | `Model.load(父)` 续训 |
| `/state` | 只读 | 上一轮保存的原生状态；首轮可能不存在 | 读投影 digest、轮次、selected |
| `/model` | 只读，仅预测时挂载 | 被选中的产物 | `predict` 读检查点 |
| `/workspace` | 可写 | 本轮工作目录 | 候选、状态副本、临时文件都写这里 |
| `/tmp` | 可写 | 临时文件（可能禁止执行程序） | 仅临时数据 |

不要假设镜像根目录可写；不要在 `/app` 下创建目录或改文件。正式评测 2 vCPU / 4 GiB / 10 GiB 临时盘。

### 2.3 一轮的完整流程

1. **收帧**：从 stdin 读 `start` 帧（dataset_key、round_index、stage_id、`task` 合同、`public_projection`、父产物与 `checkpoint_sha256`、history、`wall_seconds`、`limits`）。
2. **读数据**：按 `rnd.task["public_projection"]` 打开 train/validation；特征展开顺序严格按父检查点 `meta.paths`。
3. **复测父模型**：同一 validation 上先算父模型指标（macro-F1 / RMSE / MAE / NASA / IQR），作为本轮对照。
4. **投影 digest**：`dataset|n_train|n_val|ids|Σy`；若与上一轮相同且上一轮 selected=parent → **快进**：做 1 次 `rnd.model()`，`finish(selected_artifact_id=父产物, configuration.skipped="identical_projection")`，本轮结束（不写 state，沿用上一份）。**4.2.0 起快进只对 B/C 生效**（`CONTEST_FASTPATH` 默认 `B,C`）：A 每轮都真实重训候选、保留逐轮采纳机会，理由见技术报告 §7.7。
5. **候选训练**：按数据集分发到 `strategies/a_impl.py|b_impl.py|c_impl.py`；每个候选 `Model.load(父)` 后继续；训练预算 `max(20, min(0.40×wall, 180) − 已用时)` 秒；C 题每 25 epoch 检查墙钟，超 `budget×0.85` 早停。
6. **门槛判定**（同一 validation、与父模型配对比较）：
   - A：`gain ≥ 0.008`（n_val≥800，否则 0.020）且配对 bootstrap `P(diff>0) ≥ 0.80`；
   - B：相对改善 `≥5%` 且 `预测IQR/真值IQR ≥ 0.35`；
   - C：RMSE 改善 `≥1%` 且 NASA 不回归。
7. **上传**：过门槛 → 候选写 `/workspace` 新目录 → `submit_candidate(candidate_dir, hypothesis, change_summary, configuration, seed, checkpoint_paths, predict_argv)`；宿主校验后返回 `artifact_id` 与公开验证摘要。
8. **`finish`**：

```python
rnd.finish(
    selected_artifact_id=submitted["artifact_id"] 或 rnd.parent.artifact_id,
    hypothesis="本轮假设",
    change_summary="相对父模型实际改了什么",
    reason="根据本轮真实模型回复和平台测量做出的选择",
    configuration={"epochs": …, "selected": "candidate"|"parent", "digest": "…"},
    seed=…,
    history_ids=[item["id"] for item in history],
    model_call_ids=list(rnd.call_ids),   # 非空，且必须是本轮真实调用
)
```

9. **收尾**：只在 `round_complete → round_ack accepted=true` 后退出；stdout 只写协议 JSONL，日志走 stderr。

**硬要求**：每轮至少 1 次真实大模型调用（快进轮也一样）；`finish` 的 `model_call_ids` 必须是本轮 `rnd.model()`/传输层返回的调用号；`selected_artifact_id` 必须是本帧已知的产物（新候选或任一父/历史产物）。

### 2.4 预测

平台按选中产物的 `predict_argv` 启动进程，只挂载该产物到 `/model`，并通过 `CONTEST_ARTIFACT_PATH`/`CONTEST_ARTIFACT_MANIFEST`/`CONTEST_PREDICTION_CONTEXT` 传入上下文。stdin 每行 `{"task","inputs"}`，必须逐行输出并刷新：

```json
{"outputs":[{"label":"Ethanol","score":0.93}]}      // classification：score∈[0,1]，label 唯一，最高分唯一
{"outputs":{"value":123.4}}                          // regression：有限非负数
```

缺行、等 EOF、超时或形状不符会导致整次测量失败（平台不补值）。回归端 `max(0, logit·target_scale + target_mean)` 由官方 `Model.predict` 提供，B 题的目标重定义完全落在 `target_scale/target_mean` 上，故 `predict.py` 无需修改。

---

## 3. 镜像构建与校验

### 3.1 构建（不注入 SDK，只替换 `/app`）

```powershell
# 1) 准备打包用 /app：源码快照 + 去 __pycache__/.pyc + 最小 run.sh（LF、0755）
#    final-app/ = app-final/ 去掉 SNAPSHOT.json、去掉 __pycache__，加 run.sh
# 2) 以官方 1.0.0 镜像为底，只替换 /app，单层重打包并自校验
& $py -X utf8 repack_image.py --image <原始 harness-1.0.0 镜像 tar> --app-dir final-app --tags harness:1.0.0 --output harness-4.2.0.tar
```

> **历史构建记录（HAR-02 阶段，保留原数）**：底镜像 `harness-1.0.0-artfix.tar`（373.62 MiB，原层 `layer-759b466965a6`），
> 重打包后层为 `layer-84c872ea02fc`（373.23 MiB），保留原层载荷 13,259 个成员并逐条比对。
> 4.2.0 沿用同一 `repack_image.py` 流程（只替换 `/app`、不注入 SDK），层名与字节数以 §3.2 的 4.2.0 实测为准。

`run.sh`（平台实际执行 `python3 /app/harness.py …`，该脚本只兼容原镜像 `Entrypoint`）：

```sh
#!/bin/sh
if [ "$#" -gt 0 ]; then exec "$@"; fi
exec sleep infinity
```

### 3.2 校验

```powershell
# 结构校验 + 安全解包 + 层内 /app 卫生检查 + describe 自检
& $py -X utf8 verify_image.py --image harness-4.2.0.tar --dest image-rootfs-v420 --report runs\verify_image_v420.md
# 只校验结构、不解包
& $py -X utf8 verify_image.py --image harness-4.2.0.tar --no-extract
# 三方一致（镜像 /app vs 源码仓库 vs app_manifest.md）
& $py -X utf8 ..\ops\check_consistency.py --image harness-4.2.0.tar
```

校验项与实测结论（4.2.0）：

| 检查项 | 结果 |
|---|---|
| manifest / repositories / config 交叉一致；单层扁平；tag `harness:1.0.0` | ✅ |
| 层 sha256 == `config.rootfs.diff_ids[0]`（`3be92899b0e2…`） | ✅ |
| `history` 条数与非 empty_layer 层数一致（6 条 / 1 非空） | ✅ |
| `/app` **13 文件**与源目录逐条一致；`run.sh` 层内模式 0o755；无 `__pycache__`/`.pyc`/`contest_harness` 副本 | ✅ |
| 镜像 /app、源码仓库、`app_manifest.md` 三方一致 | ✅ **13/13，drift=0，warnings=0** |
| 成员路径无盘符、无绝对路径、无 `..`；未替换载荷逐条（大小+sha256）一致 | ✅ |
| 解包后用本机 python 在 rootfs 上跑 `describe` | ✅ 返回合法 JSON（`sdk_version:"1.0.5"`） |
| Windows 解包无法创建 822 个符号链接 | ⚠️ 本机限制，不影响 tar 本体 |

---

## 4. 本地复现

> `$py` 为本机 Python 3.12 解释器路径（示例：`C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe`），按环境替换；线程固定为 2。

### 4.1 三题离线复现（只用赛题包 public 数据）

> 以下路径相对开发工作树（`work/a|b|c`）。提交仓库中以 `evidence/` 存放冻结记录（见技术报告 §6.6），源码按 `contest_submission.py` + `strategies/` 复现同口径数字。

```powershell
$env:BASELINE_CPU_THREADS="2"; $env:OMP_NUM_THREADS="2"

# A：复现父模型 0.714030 + 漂移诊断 + 31 配置网格 + 样本外遴选 + 接口自测
& $py work\a\step1_repro.py
& $py work\a\step3_grid.py
& $py work\a\step5_recipe.py
& $py work\a\verify_strategy.py
& $py work\a\verify_xy_interface.py          # 期望输出「判定：PASS」，Δ=0.000e+00

# B：复现 681.159833 + trainer 与 Model.fit 等价 + LOBO 网格 + 交付候选体检
& $py work\b\step1_repro.py
& $py work\b\step3_verify.py
& $py work\b\step12_final.py
& $py work\b\step15_validate2.py

# C：复现 52.869/38.900 + 误差解剖 + 推荐配置/LOBO + 接口与序列化自测
cd work\c
& $py probe.py
& $py errbins.py
& $py run.py final_grid
& $py run.py lobo
& $py test_xy.py                             # 全量 / 有序截断陷阱 / stage 切换三用例
& $py test_serialize.py
```

对照期望值（容差 1%）：

| 数据集 | 父检查点（public validation） | 本版候选 | 备注 |
|---|---|---|---|
| A | macro-F1 0.714030 / acc 0.878617 | **0.755890**（+0.0419） | `verify_strategy.py` 与 `verify_xy_interface.py` 应给同一值 |
| B | RMSE 681.159833 / MAE 563.105841 | **450.316**（−33.9%） | 离线候选口径 441.00；LOBO 947→654 |
| C | RMSE 52.868573 / MAE 38.899830 | **50.258**（−4.94%） | 固定末轮 50.435；NASA 5.823e9→2.709e9 |

### 4.2 平台协议仿真（开发期工具，不随镜像提交）

真实子进程 + `subst` 挂载 + 完整 `contest.harness.workspace.v2` 帧，30 轮（3 数据集 × 10）：

```powershell
cd work\harness
$sdk105 = "<SDK 1.0.5 目录>"

# 主证据：镜像里的 /app × SDK 1.0.5 × 30 轮 × 2 线程（2c4g 口径，约 1 min）
#   先解包镜像，再以 image-verify-420\app 为被测对象
& $py -X utf8 verify_image.py --image harness-4.2.0.tar --dest image-verify-420
& $py -X utf8 platform_sim.py --app image-verify-420\app --sdk $sdk105 `
    --datasets A B C --rounds 10 --threads 2 --runs-root runs-verify-420 --no-sdk-selftest

# 对照臂：4.1.0 的"全快进"行为（A 也快进），用于验证开关与指标无损
& $py -X utf8 platform_sim.py --app image-verify-420\app --sdk $sdk105 `
    --datasets A B C --rounds 10 --threads 2 --env CONTEST_FASTPATH=A,B,C `
    --runs-root runs-verify-420-allfast --no-sdk-selftest

# 汇总检查（30 轮 PASS/FAIL + 跨运行确定性）
& $py -X utf8 verify_summary.py --run runs-verify-420 --label "默认(B,C)" `
    --run runs-verify-420-allfast --label "全快进(A,B,C)"

# 只打印一帧 start / 缩短验证集调试
& $py -X utf8 platform_sim.py --show-frame A
& $py -X utf8 platform_sim.py --datasets C --rounds 1 --val-limit 2000
```

期望：退出码 0；`runs-verify-420/summary.md` 中 A 0.755119 / B 450.316 / **C 50.867** 选中，30/30 无失败，
父检查点复现偏差 0.00%；30 轮墙钟约 52 s、峰值 RSS 229.4 MiB；两臂末轮指标与选中产物 id 逐位相同
（详见 `work/LOCAL_VERIFICATION_420.md`）。

### 4.3 单轮手动跑（不依赖仿真器）

把赛题包 `public/` 挂成 `/data`、父产物挂成 `/previous`、空目录挂成 `/workspace`，然后：

```bash
python3 /app/harness.py train < start.frame.json
python3 /app/harness.py predict < samples.jsonl
```

---

## 5. 已知限制

| # | 限制 | 影响 | 处置 |
|---|---|---|---|
| 1 | 仿真器只换 `stage_id`、不换公开投影 | 未在仿真里验证"换阶段一次性适配" | 用技术报告 §6.3 的三个模拟 stage 单独验证（首轮 −3.1%~−5.6%） |
| 2 | 仿真器模型调用是离线 stub | 未验证真实网关延迟与返回质量 | 协议层（每轮≥1 次真实调用、`model_call_ids` 非空）已验证；真实质量待试运行 |
| 3 | RSS/耗时是本机相对参考 | 不能直接当正式环境数字 | 4.2.0 在 2c4g 口径下 C 最长单轮 **20.0 s / 230 s**、峰值 **229.4 MiB / 4096 MiB**，余量充足 |
| 4 | Windows 无 chroot，只读权限未强制 | 本地写 `/app` 会"成功"，不能验证只读边界 | 代码层不写 `/app`；正式环境由平台强制 |
| 5 | 符号链接在 Windows 解包失败（822 个） | 仅影响本机解包，不影响 tar | 不影响提交；`/app` 与 site-packages 已解出并 `describe` 通过 |
| 6 | `subst` 可能残留盘符 | 下次运行冲突 | `subst` 查看后 `subst X: /D`；仿真器正常结束会自行释放 |
| 7 | B 题的 `c=1000` 是外部先验 | 隐藏集若以长寿命为主会欠预测 | 900–1,300 为平顶区（误差 ≤7%）；必要时上调 1,300–1,500（单行常量） |
| 8 | A 的 margin 0.008 按 n_val≈1,244 定 | 小验证集会偏松 | 代码已按 n_val<800 自动抬到 0.020 |

---

## 6. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `PARENT_DIGEST_MISMATCH` | `/previous` 字节被改，或绕开 `rnd.parent.path()` 直接读文件 | 只用 SDK 接口读父产物；不改 `/previous` |
| `ROUND_ACK_MISSING` | `finish` 后在 `round_ack accepted=true` 之前继续发帧或提前退出 | `finish` 后等待 ack，然后立即退出进程 |
| `MODEL_CALL_REQUIRED` | `model_call_ids` 为空或引用了非本轮调用 | 每轮先 `rnd.model()`，`finish` 用 `list(rnd.call_ids)` |
| `CHUNKED_UPLOAD_UNSUPPORTED` | 宿主未宣告 `chunked-v1`（旧宿主） | 企业赛宿主已支持分块上传；若遇旧宿主，改用 1.0.2 内联镜像 |
| `Model.save` 报目标已存在 | `/workspace` 未清空或复用路径 | 每轮用新的 `tempfile.mkdtemp()` 子目录 |
| predict 整次测量失败 | 少写一行、未 flush、等 EOF、分数越界或最高分不唯一 | 逐行输出并 flush；分类分数∈[0,1]、最高分唯一；回归非负有限 |
| `RPC_OUTCOME_UNKNOWN` | RPC 被信号/超时中断，宿主是否处理未知 | 不要捕获后继续调用；本轮会话不可复用，直接失败退出 |
| describe 未通过 | `/app/harness.py` 缺失、选手模块 import 失败、`tasks` 声明不符 | 检查 **13** 个文件齐全、`PYTHONPATH` 含 SDK、无语法错误 |

---

## 7. 复现一键清单

```powershell
# 0) 环境
$env:BASELINE_CPU_THREADS="2"; $env:OMP_NUM_THREADS="2"
$py="…\python.exe"

# 1) 三题父模型复现（应 0.00% 偏差）
& $py work\a\step1_repro.py
& $py work\b\step1_repro.py
& $py work\c\probe.py

# 2) 三题接口自测（应 PASS / 三用例全过）
& $py work\a\verify_xy_interface.py
& $py work\b\step15_validate2.py
& $py work\c\test_xy.py

# 3) 平台协议 30 轮仿真（$sdk105 见 §4.2，先 cd 到 work\harness；2 线程 = 2c4g 口径）
cd work\harness; & $py -X utf8 platform_sim.py --app image-verify-420\app --sdk $sdk105 --datasets A B C --rounds 10 --threads 2 --runs-root runs-verify-420 --no-sdk-selftest

# 4) 镜像校验 + 三方一致
& $py -X utf8 verify_image.py --image harness-4.2.0.tar --dest image-rootfs-v420 --report runs\verify_image_v420.md
& $py -X utf8 ..\ops\check_consistency.py --image harness-4.2.0.tar
```
