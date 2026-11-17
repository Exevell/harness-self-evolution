# Harness SDK 1.0.2 接入说明

## 镜像内布局

| 路径 | 作用 |
| --- | --- |
| `/app/harness.py` | 平台固定入口，原样调用 `contest_harness.cli.main` |
| `/app/contest_submission.py` | 接入层，实现 `train(rnd)` 与 `predict(sample, model_dir)` |
| `/app/baselines/` | NumPy MLP 与数据读取（可续训、无 pickle 检查点） |
| `/app/run.sh` | 无参数时常驻待命（`exec sleep infinity`），带参数时原样透传 |
| site-packages/contest_harness | 官方 SDK 1.0.2（SHA-256 `c676c8be…f85133`） |

平台以 `python3 /app/harness.py describe|train|predict` 驱动容器，因此入口不再自行启动训练；旧版"进容器就自动跑"的语义会让宿主的第一条检查命令拿不到进程，已移除。

## 一轮 train 的流程

1. 读 `rnd.task`（`dataset_key`/`task`/`metric`/`public_projection`）与 `rnd.parent` 的检查点，按 `meta["paths"]` 对齐特征顺序，缺字段补 0。
2. 读 `/data/train.jsonl`、`/data/validation.jsonl`（公开投影），先实测父模型指标作为基线。
3. 按轮次给 2–3 组 `epochs × learning_rate` 组合，各自从父检查点续训；只在公开验证集上比较，不使用隐藏测试集。
4. 调一次真实模型 `rnd.model(...)`，把实测摘要交给它，要求返回 `keep_best | try_again` 的 JSON 建议；建议合法时再补跑一组。
5. 选公开验证集指标最优的检查点写入 `/workspace/candidate-rNN/`，`rnd.submit_candidate(...)` 上传。
6. `rnd.state_workspace` + `note.json` + `rnd.save_state`。
7. `rnd.finish(...)`：`configuration` 记录 epochs/lr 与两组指标，`reason` 内嵌模型回执 `call_id`，`model_call_ids` 非空。

## predict

平台只挂载 `/model`，并给出 `CONTEST_ARTIFACT_PATH`。逐行读 stdin，写完一行再读下一行：

- 分类：输出全部标签的 `{"label","score"}`，分数裁剪到 [0,1]、保留 6 位，最高分并列时对 argmax 加 1e-6 保证唯一；
- 回归：输出 `{"value": 非负有限数}`。

## 本地验证

```bash
python3 -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v   # 官方 SDK 自测
python3 /app/harness.py describe   # 应为 protocol=contest.harness.workspace.v2 / tasks=[classification, regression]
```

另外用 SDK 自带的 JSONL 宿主替身跑通了一整轮 train（含候选上传、状态保存、`round_complete` 回执）与分类/回归两条 predict 路径。
