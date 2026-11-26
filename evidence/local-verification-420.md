# 本地验证记录 · 4.2.0（企业赛 2c4g 口径）

> 日期：2026-11-26 22:4x–22:5x
> 目的：在**不消耗任何平台试运行/正式评测次数**的前提下，确认候选镜像 `harness-4.2.0.tar`
> 的完整性、协议正确性、指标确定性、资源占用与交付物同源性，再决定是否上平台。

## 一、结论（先看这里）

| 检查 | 结果 |
|---|---|
| 镜像结构（manifest/diff_id/单层） | ✅ 通过（`verify_image.py`，11.9 s） |
| 镜像 `/app` 契约（describe 三题、SDK 1.0.5、run.sh 0755、无 `__pycache__`） | ✅ 通过 |
| 镜像 / 仓库 / 清单 三方一致 | ✅ **PASS**（13 文件，drift=0，warnings=0） |
| 2c4g 协议自评（跟着镜像 `/app` 跑，不是工作区目录） | ✅ **30/30 轮通过** |
| 指标确定性（同一代码跑 3 次，含旧 2c 记录） | ✅ 30/30 轮选中产物 id 与末轮指标**逐位相同** |
| 资源（2c4g） | ✅ 峰值 RSS **229.4 MiB / 4096 MiB（5.6%）**，最长单轮 **20.0 s / 230 s（8.7%）** |
| A 题"退出快进"改动 | ⚠️ 本地公开流上**指标中性**（见 §四），依据来自平台隐藏集证据，成本 +14.5 s/10 轮 |
| 源代码 zip 与镜像同源 | ❌→✅ **发现并已修复**：旧 zip 里 `contest_submission.py` 还是 `f14c146a`（4.1.0），已用 `work/build_source_zip.py` 重建为 `9710a252`（4.2.0），13/13 逐字节相同 |

**判定：4.2.0 达到"可上平台"的本地门槛。** 唯一未闭合项是 A 题的隐藏集恢复（本地公开流无法区分），
需要用 1 次试运行验证；正式评测前仍建议先跑这一次试运行。

## 二、跑了什么（全部本地、零平台次数）

```powershell
$py = "C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$sdk = "C:\Users\Administrator\Desktop\contest-harness-sdk-1.0.5"

# 1) 镜像完整性 + describe
cd work\harness
& $py -X utf8 verify_image.py --image .\harness-4.2.0.tar --dest .\image-verify-420

# 2) 三方 sha256 一致
cd ..
& $py -X utf8 ops\check_consistency.py --image .\harness\harness-4.2.0.tar

# 3) 2c4g 协议自评：被测 /app = 从镜像里解包出来的真实 /app
cd harness
& $py -X utf8 platform_sim.py --app .\image-verify-420\app --sdk $sdk `
    --datasets A B C --rounds 10 --threads 2 --runs-root .\runs-verify-420 --no-sdk-selftest

# 4) 对照臂：恢复 4.1.0 的"全快进"行为（A 也快进），验证快进开关确实生效且指标无损
& $py -X utf8 platform_sim.py --app .\image-verify-420\app --sdk $sdk `
    --datasets A B C --rounds 10 --threads 2 --env CONTEST_FASTPATH=A,B,C `
    --runs-root .\runs-verify-420-allfast --no-sdk-selftest

# 5) 汇总检查（新增脚本）：30 轮 × N 个运行的 PASS/FAIL + 确定性
& $py -X utf8 verify_summary.py --run .\runs-2c --label "2c 旧记录" `
    --run .\runs-verify-420 --label "镜像/app 默认(B,C)" `
    --run .\runs-verify-420-allfast --label "全快进(A,B,C)" `
    --md .\runs\verify_420_table.md --json .\runs\verify_420_table.json

# 6) 重建源代码 zip 并做同源校验
cd ..
& $py -X utf8 build_source_zip.py --repo ..\repo-harness --demo demo --submission submission `
    --out submission\项目源代码-HAR03.zip --image-app harness\image-verify-420\app
```

## 三、结果明细

### 3.1 镜像

| 项 | 值 |
|---|---|
| 文件 | `work/harness/harness-4.2.0.tar`，391,475,200 B（373.34 MiB） |
| sha256 | `e214ef6a0f282524a60cdb468e2b0ff25ed1ebe891b46978c6b1a62ea979ae75`（与模型配置一致） |
| 布局 | 单层扁平 `docker save`，1 层，diff_id 自校验通过，history 对应正确 |
| `/app` | 13 个文件 / 160.4 KiB；`run.sh` 层内 mode 0755；无 `__pycache__`/`.pyc`；无 `contest_harness` 副本 |
| describe | `{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.5","tasks":["classification","regression"]}` |

### 3.2 2c4g 协议自评（被测 `/app` 直接来自镜像）

| 数据集 | 轮数 | 采纳 | 快进轮 | 训练合计 | 最长单轮 | 峰值 RSS | 单轮预算 | 末轮指标 |
|---|---|---|---|---|---|---|---|---|
| A | 10/10 ✅ | 1（r1） | 0 | 18.0 s | 2.3 s | 45.2 MiB | 180 s | macro_f1 **0.755119** |
| B | 10/10 ✅ | 1（r1） | 9 | 6.7 s | 5.4 s | 51.9 MiB | 150 s | RMSE **450.316** |
| C | 10/10 ✅ | 1（r1） | 9 | 22.2 s | 20.0 s | 229.4 MiB | 230 s | RMSE **50.867** |
| 合计 | **30/30** | 3 | 18 | 47.0 s | — | **229.4 MiB** | — | 墙钟 **52.2 s** |

- 每轮 `model_call_ids` 非空、`finish` 选择的产物都指向本轮真实产物、`selected=parent` 时确实指向父产物；
- 上传链 `upload_begin/chunk/finish` 与候选数一致（SDK 1.0.5 分块上传）；
- 无 `round_failed`、无 RPC 乱序、父检查点指标复现偏差 0.00%；
- 首轮候选与官方基线对比：A 0.714030→0.755119、B 681.160→450.316、C 52.869→50.867。

### 3.3 对照臂（`CONTEST_FASTPATH=A,B,C`，即 4.1.0 行为）

| 数据集 | 训练合计 | 最长单轮 | 快进 | 末轮指标 |
|---|---|---|---|---|
| A | 3.5 s | 2.2 s | 9 | 0.755119（**与默认逐位相同**） |
| B | 7.3 s | 5.9 s | 9 | 450.316（相同） |
| C | 18.7 s | 17.4 s | 9 | 50.867（相同） |

→ 说明两件事：① `CONTEST_FASTPATH` 开关确实生效；② **在本地公开流上，A 快进与不快进的
逐轮选中产物、末轮指标完全相同**（30/30 轮 id 相同）。

### 3.4 确定性

同一份代码、同一 SDK、2 线程，跑 3 次（旧 2c 记录 + 镜像默认臂 + 全快进臂）：

- 三题的末轮指标三次完全一致（A 0.755119 / B 450.316218 / C 50.867442）；
- 首轮候选 id 相同（`A-r01-4a3a00cf92` / `B-r01-4e1d9f0dba` / `C-r01-f8de4c4e23`），
  30/30 轮"选中产物 id"与旧记录逐位相同。

### 3.5 源代码 zip

| 项 | 旧 zip | 新 zip |
|---|---|---|
| `contest_submission.py` | `f14c146a…`（4.1.0，A 仍在快进） | **`9710a252…`（4.2.0）** |
| `模型配置-HAR03.txt` | 5,540 B（4.1.0 版） | 5,569 B（4.2.0 版，sha `903629dd…`） |
| 成员数 / 大小 | 54 / 218,364 B | 55 / 221,461 B（+`evidence/resource-2c4g-selfeval.md`） |
| 新 zip sha256 | — | `d5f3f28873e106cd1d5b14de3d1c5de8b6cfe7c17e47d44f9cd5475ebd8b18f5` |
| 同源校验 | ❌ 与镜像不一致 | ✅ 13/13 逐字节相同 |

新增 `work/build_source_zip.py` 把"打包 + 同源校验"固化成一条命令，以后每次改代码重跑即可。

## 四、必须诚实说明的一点：A 题改动的本地证据强度

- **本地公开流无法验证 A 的隐藏集恢复**：A 快进与不快进在本地 30 轮里选中的产物完全相同，
  所以本地指标零差异——这不是"改动有效"的证据，只是"改动不会退化公开指标"；
- 改动依据是平台隐藏集：v12（每轮重训）A **0.9021** vs v14（A 快进）**0.8828**；
- 代价：2c4g 下 A 训练 3.5 s → 18.0 s（10 轮），整体墙钟 35.0 s → 52.2 s；
  但墙钟由 C 独占，真正瓶颈单轮仍只有 20 s / 230 s，**代价可接受**；
- 因此建议：**用 1 次试运行确认 A 回到 ≈0.90 且 2c 计时/Token 正常，再走正式评测。**

## 五、遗留项（不影响本地验证结论，但影响最终交付）

1. `submission/提交清单-HAR03.md` 与 `docs/materials_index.md` 仍写 **4.1.0 / d6f0f284… / 207 KB**，
   需改成 4.2.0 / `e214ef6a…` / 216 KB；
2. `项目说明书.pdf`、`系统运行说明文档.pdf`、`答辩PPT-HAR04.pptx` 生成于 4.2.0 之前，
   正文里的镜像版本与"资源补丁"章节需要同步 4.2.0（并把 §7.6 的 A 快进结论补一条"HAR-04.4 已退出快进"）；
3. 网盘上的镜像仍是 3.0.2，4.2.0 需要重新上传并更新 `镜像网盘链接.txt`（需队伍操作网盘）。
