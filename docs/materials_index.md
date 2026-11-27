# 作品文档 · HAR-04（队伍 070228 · 北京印刷学院 · 镜像 `harness-4.2.0.tar`）

> 本文档包含四部分：**材料清单与核验索引**、**技术报告（≤10 页）**、**来源声明**、**系统运行说明**。
> 平台没有单独入口的材料（源码仓库、云端实验记录、演示视频）均在本节以链接方式提供。

## 0. 材料清单与核验索引（评委看这里）

| 材料 | 位置 / 链接 | 说明 |
|---|---|---|
| **系统镜像** | **`harness-4.2.0.tar`**（linux/amd64，单层扁平）<br>大小 **391,475,200 B**（373.34 MiB）<br>sha256 **`e214ef6a0f282524a60cdb468e2b0ff25ed1ebe891b46978c6b1a62ea979ae75`**<br>tag `harness:1.0.0`；内置 SDK `contest_harness 1.0.5`；`/app` 13 文件 / 160.4 KiB | 已通过本地镜像自检与"镜像 / 仓库 / 清单"三方一致（13/13）；平台归档与网盘链接由队伍在提交时同步（网盘当前仍为历史 3.0.2 镜像，4.2.0 编号以提交回执为准）。入口 `/app/run.sh` → `python3 /app/harness.py describe\|train\|predict` |
| **源码仓库** | <https://github.com/Exevell/harness-self-evolution>（main 分支） | 含 `contest_submission.py`、`strategies/`、`baselines/`、`docs/`、`evidence/`（三题实验报告 + 30 轮协议仿真记录）、`tools/`（`build_source_zip.py` 打包+同源校验、`verify_summary.py` 30 轮汇总）。镜像内 `/app` 与仓库文件**逐字节一致（13/13）**，可用 `tools/build_source_zip.py --image-app` 一键复核；sha256 见文末附录 |
| **本地验证记录** | `work/LOCAL_VERIFICATION_420.md`（仓库 `evidence/local-verification-420.md`） | 4.2.0 镜像完整性、三方一致、**2c4g 30/30 轮协议自评**、三次运行确定性（30/30 选中产物 id 逐位相同）、源码 zip 同源校验；数字与技术报告 §7.1 表 B / §7.7 一致 |
| **云端实验记录（SwanLab）** | <https://swanlab.cn/@070228Ggx/harness-self-evolution/runs/wfq2u9mz> | 含三数据集逐轮候选、指标、决策理由、隐藏测试指标与资源计量；**其中至少 3 次完整的"读历史—做实验—依据结果决策"记录**：<br>① 第 1 轮：父模型 0.714030 → 候选 0.755119，采纳；<br>② 第 3 轮：stage-02 漂移后父模型 0.739598 → 候选 0.946537，采纳；<br>③ 第 7 轮：stage-05 父模型掉到 0.828515 → 候选 0.985285，单轮恢复并采纳 |
| **演示视频（≤5 分钟）** | [网盘链接](https://pan.baidu.com/s/1lHphExR9QXfasV8YRRr6Yw)（提取码 `sxsi`） | 演示 `describe → train → predict` 全流程：第 1 轮父模型 0.714030→候选 0.755119 被采纳；第 2 轮候选未超门槛，保留父模型；以及平台十轮逐轮记录与 SwanLab 实验记录 |
| **离线演示程序** | `demo/run_demo.py`（源代码包内，双击 `demo/演示-双击运行.cmd`） | 同上流程的本地一键复现，任何有 Python+NumPy 的机器可离线运行 |
| **技术报告（≤10 页）** | 本文档第 1 部分 | 赛题理解、系统架构、漂移归因、三题方法、实验设计与结果、资源效率、局限 |
| **设计说明** | 本文档第 2 部分 | 总体架构（两层受控 RSI）、关键设计决策与被否决的备选方案、安全可控性、可复现性 |
| **来源声明** | 本文档第 3 部分 | 数据/模型/组件/SDK 来源与 12 条合规声明 |
| **系统运行说明** | 本文档第 4 部分 | 平台侧运行、镜像构建与校验、本地复现、已知限制 |
| **模型调用信息** | 参赛模型 `Qwen/Qwen3.5-4B`（平台白名单内，`base_model_confirmed=true`） | 调用经平台宿主网关（`host_gateway_rpc`），PyTRiO Key 只填在平台密钥栏，**不写入镜像/源码/文档**。每轮至少 1 次真实调用，回执号记录在每轮 `finish` 的 `model_call_ids` |
| **数据使用** | 仅官方赛题包 `public/train.jsonl`、`public/validation.jsonl` | 未使用任何原始 CSV、隐藏轴承、隐藏测试标签；验证集只用于测量与选型，不进入梯度训练 |

**给核验同学的一句话**：镜像 sha256、仓库、SwanLab run、演示视频四者一一对应；`/app` 内每个文件的 sha256 在文末附录，可逐字节核对。

---
