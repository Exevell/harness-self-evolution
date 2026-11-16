#!/bin/bash
# ============================================================
# 工业模型自进化 Harness - 一键运行入口
# ============================================================
#
# 评测平台调用此脚本启动自进化系统。
# 系统须无人工干预地完成全部阶段并按时输出结果。
#
# 环境变量（由平台注入）:
#   TASK          - 赛题任务 A/B/C（默认 B）
#   ROUNDS        - 自进化轮数（默认 10）
#   CHECKPOINT    - 基线检查点路径
#   TRAIN_JSONL   - 训练集 JSONL 路径
#   VAL_JSONL     - 验证集 JSONL 路径
#   OUTPUT_DIR    - 输出目录（平台指定）
#   PYTHON        - Python 解释器路径（默认 python3）
#
# 退出码:
#   0 - 自进化完成
#   1 - 运行失败

set -euo pipefail

TASK="${TASK:-B}"
ROUNDS="${ROUNDS:-10}"
CHECKPOINT="${CHECKPOINT:-}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
VAL_JSONL="${VAL_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/harness_output}"
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo "工业模型自进化 Harness v1.0.0"
echo "============================================================"
echo "任务:     ${TASK}"
echo "轮数:     ${ROUNDS}"
echo "检查点:   ${CHECKPOINT}"
echo "训练集:   ${TRAIN_JSONL}"
echo "验证集:   ${VAL_JSONL}"
echo "输出目录: ${OUTPUT_DIR}"
echo "Python:   ${PYTHON}"
echo "============================================================"
echo ""

# 1. 验证输入
if [ ! -f "${CHECKPOINT}" ]; then
    echo "错误: 检查点不存在: ${CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${TRAIN_JSONL}" ]; then
    echo "错误: 训练集不存在: ${TRAIN_JSONL}" >&2
    exit 1
fi

if [ ! -f "${VAL_JSONL}" ]; then
    echo "错误: 验证集不存在: ${VAL_JSONL}" >&2
    exit 1
fi

# 2. 定位 baselines 目录（model.py 与 configs 所在位置）
BASELINES="${BASELINES_DIR:-/app/tasks/${TASK}/baselines}"

if [ ! -f "${BASELINES}/model.py" ]; then
    echo "错误: 未找到 baselines 目录: ${BASELINES}" >&2
    exit 1
fi

export HARNESS_BASELINES="${BASELINES}"

# 3. 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 4. 运行自进化
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[启动] 自进化 Harness..."
echo ""

${PYTHON} "${SCRIPT_DIR}/harness/self_evolution.py" \
    --task "${TASK}" \
    --rounds "${ROUNDS}" \
    --checkpoint "${CHECKPOINT}" \
    --train "${TRAIN_JSONL}" \
    --validation "${VAL_JSONL}" \
    --output "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "自进化完成"
    echo "输出目录: ${OUTPUT_DIR}"
    echo "实验记录: ${OUTPUT_DIR}/experiment_history.jsonl"
else
    echo "自进化失败 (exit code: ${EXIT_CODE})" >&2
fi
echo "============================================================"

exit ${EXIT_CODE}
