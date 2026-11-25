# HAR-02 reference image definition.
#
# The submitted image (harness-2.0.2.tar) was produced as a flat, single-layer
# docker archive whose /app tree is byte-identical to this repository.  It is
# reproducible with either:
#   (a) `docker build -f Dockerfile -t harness:1.0.0 .`  (this file), or
#   (b) repacking the flat archive and replacing /app (no Docker required).
#
# Runtime contract (platform side):
#   python3 /app/harness.py describe | train | predict
# `/app`, `/data`, `/previous`, `/state`, `/model` are read-only; `/workspace`
# and `/tmp` are writable.
FROM python:3.12.4-slim

LABEL org.opencontainers.image.title="HAR-02 industrial-model self-evolution harness"
LABEL org.opencontainers.image.version="2.0.2"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    BASELINE_CPU_THREADS=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    CONTEST_SUBMISSION=contest_submission

WORKDIR /app

# NumPy is the only third-party dependency of the training/prediction code.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Participant SDK (contest_harness, protocol contest.harness.workspace.v2).
# Keep the version the platform ships with; do not pin a newer one blindly --
# 1.0.3+ requires a chunked-upload host.
COPY sdk/contest_harness /tmp/sdk/contest_harness
COPY sdk/pyproject.toml /tmp/sdk/pyproject.toml
RUN pip install --no-cache-dir /tmp/sdk && rm -rf /tmp/sdk

COPY baselines/ /app/baselines/
COPY strategies/ /app/strategies/
COPY contest_submission.py harness.py stage_policy.py run.sh /app/
COPY docs/ /app/docs/

RUN chmod 0755 /app/run.sh

ENTRYPOINT ["/app/run.sh"]
