FROM python:3.12.4-slim

LABEL maintainer="参赛团队"
LABEL description="工业模型自进化 Harness 系统镜像"
LABEL version="1.0.0"

# 环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# 工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt /app/

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY harness/ /app/harness/
COPY shared/ /app/shared/
COPY tasks/ /app/tasks/
COPY run_harness.py /app/
COPY run.sh /app/
COPY upload_to_swanlab.py /app/

# 复制文档（技术报告、设计论证、运行说明、合规声明）
COPY docs/ /app/docs/
COPY README.md REPRODUCE.md SUBMISSION_CHECKLIST.md /app/

# 确保脚本可执行
RUN chmod +x /app/run.sh

# 数据挂载点（由平台注入）
VOLUME ["/data/train", "/data/validation", "/data/checkpoint", "/output"]

# 默认环境变量
ENV TASK=B \
    ROUNDS=10 \
    CHECKPOINT=/data/checkpoint/checkpoint.npz \
    TRAIN_JSONL=/data/train/train.jsonl \
    VAL_JSONL=/data/validation/validation.jsonl \
    OUTPUT_DIR=/output

# 入口
ENTRYPOINT ["/app/run.sh"]
