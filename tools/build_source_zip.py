#!/usr/bin/env python3
"""build_source_zip.py -- 从仓库/工作区重建"项目源代码"交付 zip，并做与镜像 /app 的同源校验。

为什么要有这个脚本：
  本地验证发现 `work/submission/项目源代码-HAR03.zip` 里的 `contest_submission.py`
  sha256=f14c146a…，而 4.2.0 镜像 /app 里是 9710a252…（A 题退出快进后的版本）——
  zip 落后于镜像，会直接违反"镜像 /app、源码包、文档附录同一份代码"的承诺。
  之前是手工打包，容易忘；这里固化成一条命令。

用法：
  python build_source_zip.py \
      --repo repo-harness --demo work/demo --submission work/submission \
      --out work/submission/项目源代码-HAR03.zip \
      --image-app work/harness/image-verify-420/app

校验：zip 内每个 app 源文件（contest_submission.py / harness.py / run.sh /
stage_policy.py / strategies/* / baselines/*）必须与 `--image-app` 逐字节相同，
否则退出码 1。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".idea", ".vscode", "_run"}
SKIP_SUFFIX = {".pyc", ".pyo"}
APP_FILES = [
    "contest_submission.py", "harness.py", "run.sh", "stage_policy.py",
    "strategies/__init__.py", "strategies/core.py", "strategies/a_impl.py",
    "strategies/b_impl.py", "strategies/c_impl.py",
    "baselines/__init__.py", "baselines/data.py", "baselines/model.py", "baselines/predict.py",
]
# 提交文档：从 work/submission 平铺到 zip 根（与旧版 zip 布局一致）
SUBMISSION_DOCS = [
    "开源组件与许可证.md", "提交清单-HAR03.md", "模型配置-HAR03.txt",
    "演示视频链接.txt", "镜像网盘链接.txt",
]


def walk(root: Path, arc_prefix: str = ""):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        yield p, (arc_prefix + rel.as_posix())


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="repo-harness")
    ap.add_argument("--demo", default="work/demo")
    ap.add_argument("--submission", default="work/submission")
    ap.add_argument("--out", default="work/submission/项目源代码-HAR03.zip")
    ap.add_argument("--image-app", default=None, help="镜像解包后的 /app 目录，用于同源校验")
    args = ap.parse_args(argv)

    repo = Path(args.repo).resolve()
    demo = Path(args.demo).resolve()
    submission = Path(args.submission).resolve()
    out = Path(args.out).resolve()
    if not repo.is_dir():
        sys.exit("repo not found: %s" % repo)

    entries = []
    entries += list(walk(repo))
    entries += list(walk(demo, "demo/"))
    for name in SUBMISSION_DOCS:
        p = submission / name
        if p.exists():
            entries.append((p, name))
        else:
            print("WARN: 缺失提交文档 %s" % p)

    seen, dup = set(), []
    for _, arc in entries:
        if arc in seen:
            dup.append(arc)
        seen.add(arc)
    if dup:
        sys.exit("重复成员: %s" % dup)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    manifest = {}
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path, arc in entries:
            data = path.read_bytes()
            manifest[arc] = sha(data)
            z.writestr(arc, data)
    tmp.replace(out)

    size = out.stat().st_size
    print("zip: %s (%d 文件, %d 字节 = %.1f KB)" % (out, len(entries), size, size / 1024))

    bad = []
    if args.image_app:
        app = Path(args.image_app).resolve()
        for rel in APP_FILES:
            img = app / rel
            if not img.exists():
                bad.append("%s: 镜像缺少" % rel)
                continue
            if sha(img.read_bytes()) != manifest.get(rel):
                bad.append("%s: zip=%s 镜像=%s" % (rel, (manifest.get(rel) or "缺失")[:16], sha(img.read_bytes())[:16]))
        print("同源校验（zip vs 镜像 /app）：%s" % ("PASS（13/13 逐字节相同）" if not bad else "FAIL"))
        for b in bad:
            print(" -", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
