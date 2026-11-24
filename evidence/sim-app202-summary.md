# platform_sim 十轮压测 summary

生成时间：2026-11-24 18:53:45

- 协议：`contest.harness.workspace.v2`，prompt_version=`industrial-workspace-flexible-v3`
- 宿主：AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD / 12 cores，Python `3.12.14`，平台替身 `work/harness/platform_sim.py`
- SDK 副本（只读引用）：`C:\Users\Administrator\Desktop\管侠3--涛妈之战\20261115-235804-4e9d4ddb-90e0-4a41-a17e-34cdc92482b0\sdk\contest-harness-sdk-1.0.2`
- 被测 /app：`C:\Users\Administrator\Documents\deepseek-harness\default-workspace\work\harness\app-202`
- 每轮预算 wall_seconds：A=180s, B=150s, C=230s
- predict 校验：完整 public/validation.jsonl
- 并发：train 与 predict 串行，任何时刻最多 1 个子进程；OMP/BLAS 线程=2

## 数据集 A（classification，主指标 macro_f1，越高更好）

describe：`{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}`

父检查点（官方基线）复现：`baseline-A-v1`，sha256=`sha256:fd1e08b2aad08b2e…`，公开验证 {'macro_f1': 0.71403, 'accuracy': 0.878617, 'rows': 1244}（predict 0.8s / 峰值 RSS 32.7 MiB）

基线一致性（容忍 1%）：✅ macro_f1=0.71403（期望 0.71403，偏差 0.00%）；✅ accuracy=0.878617（期望 0.878617，偏差 0.00%）

| 轮 | stage | train wall(s) | train 峰值RSS | 测量 wall(s) | 候选字节 | 选中 | macro_f1 | 父模型 | 候选 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A-stage-01 | 3.50 | 45.9 MiB | 0.88 | 129.1 KiB | candidate | 0.755119 | 0.71403 | 0.755119 |
| 2 | A-stage-02 | 2.42 | 45.1 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 3 | A-stage-03 | 0.58 | 38.7 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 4 | A-stage-04 | 0.56 | 38.7 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 5 | A-stage-05 | 0.55 | 38.9 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 6 | A-stage-06 | 0.59 | 38.6 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 7 | A-stage-06 | 0.61 | 38.8 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 8 | A-stage-07 | 0.59 | 39.1 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 9 | A-stage-07 | 0.59 | 39.3 MiB | - | - | parent | 0.755119 | 0.755119 | - |
| 10 | A-stage-08 | 0.58 | 38.9 MiB | - | - | parent | 0.755119 | 0.755119 | - |

- 合计：train 10.6s + 测量 0.9s = 11.5s（10 轮，均值 1.1s/轮，最长单轮 train 3.5s）
- 所有子进程峰峰值 RSS：45.9 MiB；候选总字节：129.1 KiB；保留父模型：9/10 轮；数据集总耗时 11.8s
- 协议帧统计：candidate_request=1, history_request=10, model_request=10, state_request=2
- predict 协议校验：全部通过（1 次候选测量；每行 1 条输出、分数∈[0,1]、最高分唯一）（保留父模型 9/10 轮）
- 官方 prompt 被重复塞进消息的轮数：0/10

## 数据集 B（regression，主指标 rmse，越低更好）

describe：`{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}`

父检查点（官方基线）复现：`baseline-B-v1`，sha256=`sha256:c7d819fcee9d2545…`，公开验证 {'rmse': 681.159833, 'mae': 563.105841, 'bias': -365.383949, 'rows': 1679}（predict 0.6s / 峰值 RSS 31.2 MiB）

基线一致性（容忍 1%）：✅ rmse=681.160（期望 681.160，偏差 0.00%）；✅ mae=563.106（期望 563.106，偏差 0.00%）

| 轮 | stage | train wall(s) | train 峰值RSS | 测量 wall(s) | 候选字节 | 选中 | rmse | 父模型 | 候选 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B-stage-01 | 5.72 | 51.4 MiB | 0.55 | 35.6 KiB | candidate | 450.316 | 681.160 | 450.316 |
| 2 | B-stage-01 | 5.17 | 51.0 MiB | - | - | parent | 450.316 | 450.316 | - |
| 3 | B-stage-01 | 1.08 | 43.4 MiB | - | - | parent | 450.316 | 450.316 | - |
| 4 | B-stage-01 | 1.09 | 43.0 MiB | - | - | parent | 450.316 | 450.316 | - |
| 5 | B-stage-02 | 1.06 | 43.4 MiB | - | - | parent | 450.316 | 450.316 | - |
| 6 | B-stage-02 | 1.06 | 43.2 MiB | - | - | parent | 450.316 | 450.316 | - |
| 7 | B-stage-02 | 1.05 | 43.1 MiB | - | - | parent | 450.316 | 450.316 | - |
| 8 | B-stage-03 | 1.08 | 43.4 MiB | - | - | parent | 450.316 | 450.316 | - |
| 9 | B-stage-03 | 1.08 | 43.4 MiB | - | - | parent | 450.316 | 450.316 | - |
| 10 | B-stage-03 | 1.08 | 43.4 MiB | - | - | parent | 450.316 | 450.316 | - |

- 合计：train 19.5s + 测量 0.5s = 20.0s（10 轮，均值 2.0s/轮，最长单轮 train 5.7s）
- 所有子进程峰峰值 RSS：51.4 MiB；候选总字节：35.6 KiB；保留父模型：9/10 轮；数据集总耗时 20.3s
- 协议帧统计：candidate_request=1, history_request=10, model_request=10, state_request=2
- predict 协议校验：全部通过（1 次候选测量；每行 1 条输出、非负有限数）（保留父模型 9/10 轮）
- 官方 prompt 被重复塞进消息的轮数：0/10

## 数据集 C（regression，主指标 rmse，越低更好）

describe：`{"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.2","tasks":["classification","regression"]}`

父检查点（官方基线）复现：`baseline-C-v1`，sha256=`sha256:98ec6e30e38aee56…`，公开验证 {'rmse': 52.868573, 'mae': 38.89983, 'bias': 1.078266, 'rows': 23049}（predict 4.0s / 峰值 RSS 30.8 MiB）

基线一致性（容忍 1%）：✅ rmse=52.869（期望 52.869，偏差 0.00%）；✅ mae=38.900（期望 38.900，偏差 0.00%）

| 轮 | stage | train wall(s) | train 峰值RSS | 测量 wall(s) | 候选字节 | 选中 | rmse | 父模型 | 候选 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | C-stage-01 | 59.86 | 229.0 MiB | 4.86 | 28.2 KiB | candidate | 50.258 | 52.869 | 50.258 |
| 2 | C-stage-01 | 60.36 | 228.3 MiB | - | - | parent | 50.258 | 50.258 | - |
| 3 | C-stage-02 | 8.16 | 182.0 MiB | - | - | parent | 50.258 | 50.258 | - |
| 4 | C-stage-02 | 8.05 | 181.5 MiB | - | - | parent | 50.258 | 50.258 | - |
| 5 | C-stage-02 | 8.05 | 182.1 MiB | - | - | parent | 50.258 | 50.258 | - |
| 6 | C-stage-03 | 8.42 | 181.3 MiB | - | - | parent | 50.258 | 50.258 | - |
| 7 | C-stage-03 | 8.17 | 182.1 MiB | - | - | parent | 50.258 | 50.258 | - |
| 8 | C-stage-03 | 8.19 | 181.5 MiB | - | - | parent | 50.258 | 50.258 | - |
| 9 | C-stage-04 | 8.39 | 182.0 MiB | - | - | parent | 50.258 | 50.258 | - |
| 10 | C-stage-04 | 8.28 | 182.0 MiB | - | - | parent | 50.258 | 50.258 | - |

- 合计：train 185.9s + 测量 4.9s = 190.8s（10 轮，均值 19.1s/轮，最长单轮 train 60.4s）
- 所有子进程峰峰值 RSS：229.0 MiB；候选总字节：28.2 KiB；保留父模型：9/10 轮；数据集总耗时 190.3s
- 协议帧统计：candidate_request=1, history_request=10, model_request=10, state_request=2
- predict 协议校验：全部通过（1 次候选测量；每行 1 条输出、非负有限数）（保留父模型 9/10 轮）
- 官方 prompt 被重复塞进消息的轮数：0/10

## 结论

- 3 个数据集 × 10 轮 = 30 轮全部通过：无 round_failed、无 RPC 乱序、无上传摘要错误、每轮 finish 都引用了本轮真实 model 调用。
- 子进程 wall time：train 216.0s + predict 测量 6.3s = 222.3s。
- 最长单轮 train：60.4s；每轮预算参考 {'A': 180, 'B': 150, 'C': 230}。

