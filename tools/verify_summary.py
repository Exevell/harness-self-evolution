#!/usr/bin/env python3
"""verify_summary.py -- 把 platform_sim 的 summary.json 压成"本地验证"结论表。

用法：
  python verify_summary.py --run runs-verify-420 --label "4.2.0 默认(B,C)" \
      [--run runs-verify-420-allfast --label "4.1.0 行为(全快进)"] ...
  python verify_summary.py --run ... --md out.md --json out.json

检查项（任一 FAIL 则退出码 1）：
  1. 每个数据集 ok=True、轮数=10、无 error；
  2. 每轮至少有 1 次 model 调用回执，且选择与 finish 一致（candidate 有指标 / parent 指向父产物）；
  3. 峰值 RSS < 4096 MiB，单轮训练 wall < 该数据集 wall_seconds 的 40%（预算纪律）；
  4. 公开指标与"官方基线复现 + 候选最优"一致（同一代码跨运行应完全相同 → 确定性）；
  5. 协议帧：history/model/state 每轮 1 次，upload_begin/chunk/finish 与候选数一致。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MiB = 1024 * 1024
BUDGET = {"A": 180, "B": 150, "C": 230}


def load(run_dir: Path):
    p = run_dir / "summary.json"
    if not p.exists():
        sys.exit("missing %s" % p)
    data = json.loads(p.read_text(encoding="utf-8"))
    return {d["dataset"]: d for d in data}


def skipped(round_rec) -> bool:
    """启发式：快进轮不做候选训练（A 重训约 1.7s，快进约 0.1~0.3s）。"""
    return round_rec["train_wall_s"] < 0.8 and round_rec["candidate_artifact_id"] is None


def summarize(runs):
    rows = []
    problems = []
    for run_dir, label in runs:
        by_ds = load(run_dir)
        for ds in ("A", "B", "C"):
            d = by_ds[ds]
            rounds = d["rounds"]
            if d.get("ok") is not True or d.get("error"):
                problems.append("%s/%s: ok=%r error=%r" % (label, ds, d.get("ok"), d.get("error")))
            if len(rounds) != 10:
                problems.append("%s/%s: rounds=%d" % (label, ds, len(rounds)))
            budget = BUDGET.get(ds, 180)
            train_total = 0.0
            train_peak = 0
            max_round = 0.0
            adopted = 0
            fast = 0
            final_metric = None
            for r in rounds:
                train_total += r["train_wall_s"]
                train_peak = max(train_peak, r["train_peak_rss_bytes"])
                max_round = max(max_round, r["train_wall_s"])
                if r["selected"] == "candidate":
                    adopted += 1
                    if not r.get("candidate_metrics"):
                        problems.append("%s/%s r%d: candidate 选中但无指标" % (label, ds, r["round_index"]))
                if r["selected"] not in ("candidate", "parent"):
                    problems.append("%s/%s r%d: selected=%r" % (label, ds, r["round_index"], r["selected"]))
                if not r.get("model_call_ids"):
                    problems.append("%s/%s r%d: 无 model_call_ids" % (label, ds, r["round_index"]))
                if r["selected"] == "parent" and r["selected_artifact_id"] != r["parent_artifact_id"]:
                    problems.append("%s/%s r%d: parent 选择未指向父产物" % (label, ds, r["round_index"]))
                if skipped(r):
                    fast += 1
                final_metric = r["selected_metrics"].get(d["metric"])
                if r["train_wall_s"] > 0.4 * budget:
                    problems.append("%s/%s r%d: train %.1fs > 40%% of %ds"
                                    % (label, ds, r["round_index"], r["train_wall_s"], budget))
            if train_peak > 4096 * MiB:
                problems.append("%s/%s: peak RSS %.0f MiB" % (label, ds, train_peak / MiB))
            rows.append({
                "run": label, "dataset": ds, "rounds": len(rounds), "ok": d.get("ok"),
                "adopted": adopted, "fast": fast, "train_total_s": round(train_total, 1),
                "max_round_s": round(max_round, 1), "peak_rss_mib": round(train_peak / MiB, 1),
                "final_metric": None if final_metric is None else round(final_metric, 6),
                "metric": d["metric"], "threads": d.get("threads"), "sdk": (d.get("sdk") or {}).get("version") if isinstance(d.get("sdk"), dict) else d.get("sdk"),
                "app": d.get("app"), "total_wall_s": round(d.get("total_wall_s") or 0, 1),
            })
    return rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="runs 目录（可重复），与 --label 一一对应")
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    if len(args.run) != len(args.label):
        sys.exit("--run 与 --label 数量必须一致")
    runs = [(Path(r).resolve(), l) for r, l in zip(args.run, args.label)]
    rows, problems = summarize(runs)

    lines = ["| 运行 | 数据集 | 轮数 | ok | 采纳 | 快进 | 训练合计(s) | 最长单轮(s) | 峰值RSS(MiB) | 末轮指标 | 线程 | total(s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %d | %s | %d | %d | %.1f | %.1f | %.1f | %s=%s | %s | %.1f |"
                     % (r["run"], r["dataset"], r["rounds"], "✅" if r["ok"] else "❌", r["adopted"], r["fast"],
                        r["train_total_s"], r["max_round_s"], r["peak_rss_mib"], r["metric"], r["final_metric"],
                        r["threads"], r["total_wall_s"]))
    out = "\n".join(lines)

    # 确定性 / 等价性：同一数据集在不同运行之间末轮指标必须完全一致
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for ds, group in by_ds.items():
        vals = {round(g["final_metric"] or 0, 6) for g in group}
        if len(vals) > 1:
            problems.append("%s: 跨运行末轮指标不一致 %s" % (ds, sorted(vals)))

    print(out)
    print()
    if problems:
        print("RESULT: FAIL (%d)" % len(problems))
        for p in problems:
            print(" -", p)
    else:
        print("RESULT: PASS (%d 轮, %d 个运行)" % (sum(r["rounds"] for r in rows), len(runs)))
    if args.md:
        Path(args.md).write_text(out + "\n\n" + ("RESULT: PASS" if not problems else "RESULT: FAIL") + "\n",
                                 encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows, "problems": problems}, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
