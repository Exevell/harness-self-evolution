#!/bin/bash
# ============================================================
# 工业模型自进化 Harness - 运行入口
# ============================================================
#
# 两种被调用方式：
#   1) 不带参数：作为镜像 ENTRYPOINT，按环境变量跑自进化。
#   2) 带参数：作为容器内命令执行器（平台用 `bash -lc ...` 检查容器时，
#      参数会附加到 ENTRYPOINT 之后，这里直接 exec 平台命令，不做任何拦截）。
#
# 环境变量（平台注入，均可不填，缺省时自动探测）：
#   TASK / ROUNDS / CHECKPOINT / TRAIN_JSONL / VAL_JSONL / OUTPUT_DIR
#   BASELINES_DIR / PYTHON / PYTRIO_API_KEY / PYTRIO_MODEL / HARNESS_LLM

set -uo pipefail

log() { echo "[harness] $*"; }

# ---------- 0) 命令透传 ----------
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

TASK="${TASK:-B}"
ROUNDS="${ROUNDS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/harness_output}"
PYTHON="${PYTHON:-python3}"

# 找第一个存在的文件
first_file() {
    for c in "$@"; do
        if [ -n "${c}" ] && [ -f "${c}" ]; then
            printf '%s' "${c}"
            return 0
        fi
    done
    return 1
}

# ---------- 1) 定位输入（平台挂载 /previous、/data 或显式环境变量） ----------
CHECKPOINT="${CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
    CHECKPOINT="$(first_file \
        /previous/checkpoint.npz \
        /previous/checkpoint/checkpoint.npz \
        /data/checkpoint/checkpoint.npz \
        /checkpoint/checkpoint.npz \
        || find /previous /data /checkpoint -maxdepth 4 -name 'checkpoint.npz' 2>/dev/null | head -1)"
fi

TRAIN_JSONL="${TRAIN_JSONL:-}"
if [ -z "${TRAIN_JSONL}" ] || [ ! -f "${TRAIN_JSONL}" ]; then
    TRAIN_JSONL="$(first_file \
        /data/train.jsonl \
        /data/train/train.jsonl \
        /previous/train.jsonl \
        /previous/public/train.jsonl \
        || find /data /previous -maxdepth 4 -name 'train*.jsonl' 2>/dev/null | head -1)"
fi

VAL_JSONL="${VAL_JSONL:-}"
if [ -z "${VAL_JSONL}" ] || [ ! -f "${VAL_JSONL}" ]; then
    VAL_JSONL="$(first_file \
        /data/validation.jsonl \
        /data/validation/validation.jsonl \
        /previous/validation.jsonl \
        /previous/public/validation.jsonl \
        || find /data /previous -maxdepth 4 -name 'validation*.jsonl' 2>/dev/null | head -1)"
fi

BASELINES="${BASELINES_DIR:-}"
if [ -z "${BASELINES}" ] || [ ! -f "${BASELINES}/model.py" ]; then
    BASELINES="$(first_file \
        "/app/tasks/${TASK}/baselines/model.py" \
        || true)"
    if [ -n "${BASELINES}" ]; then
        BASELINES="$(dirname "${BASELINES}")"
    else
        BASELINES="$(find /previous /app/tasks /data -maxdepth 5 -name 'model.py' 2>/dev/null | head -1)"
        [ -n "${BASELINES}" ] && BASELINES="$(dirname "${BASELINES}")"
    fi
fi

log "任务=${TASK} 轮数=${ROUNDS}"
log "检查点=${CHECKPOINT:-<未找到>}"
log "训练集=${TRAIN_JSONL:-<未找到>}"
log "验证集=${VAL_JSONL:-<未找到>}"
log "baselines=${BASELINES:-<未找到>}"
log "输出目录=${OUTPUT_DIR}"

# ---------- 2) 输入不全：保持容器存活，便于平台检查，不空转跑坏 ----------
if [ -z "${CHECKPOINT}" ] || [ -z "${TRAIN_JSONL}" ] || [ -z "${VAL_JSONL}" ] || [ -z "${BASELINES}" ]; then
    log "输入未就绪，容器进入待命（sleep infinity），不使用任何算力。"
    exec sleep infinity
fi

export HARNESS_BASELINES="${BASELINES}"
mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- 3) 跑自进化 ----------
log "启动自进化..."
"${PYTHON}" "${SCRIPT_DIR}/harness/self_evolution.py" \
    --task "${TASK}" \
    --rounds "${ROUNDS}" \
    --checkpoint "${CHECKPOINT}" \
    --train "${TRAIN_JSONL}" \
    --validation "${VAL_JSONL}" \
    --output "${OUTPUT_DIR}"
EXIT_CODE=$?

log "退出码=${EXIT_CODE}"
log "实验记录=${OUTPUT_DIR}/experiment_history.jsonl"
exit ${EXIT_CODE}
