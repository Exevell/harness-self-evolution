#!/bin/bash
# ============================================================
# HAR-01 Harness 镜像入口（平台 worker 模式）
# ============================================================
#
# 平台的用法是：容器常驻，然后由宿主在容器内执行
#   python3 /app/harness.py describe
#   python3 /app/harness.py train
#   python3 /app/harness.py predict
# 因此本入口只做两件事：
#   1) 带参数时原样透传并执行（平台任何检查命令都不会被拦截）；
#   2) 不带参数时常驻待命，不占算力、不退出，避免容器秒退。
#
# 需要本地/离线自跑时，显式使用：
#   python3 /app/run_harness.py --task B --rounds 10 ...
set -uo pipefail

log() { echo "[harness] $*"; }

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

log "worker 待命：平台通过 /app/harness.py 驱动 describe / train / predict。"
log "python=$(command -v python3 || echo 未找到)  工作目录=$(pwd)"

# 常驻，不消耗 CPU
exec sleep infinity
