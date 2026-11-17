"""平台固定入口：/app/harness.py。平台会追加 describe / train / predict。"""
import sys

from contest_harness.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
