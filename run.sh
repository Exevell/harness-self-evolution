#!/bin/sh
# Entrypoint of the HAR-02 harness image.
#
#   * with arguments (the platform appends `python3 /app/harness.py describe|train|predict`)
#     the command is executed verbatim;
#   * without arguments the container stays alive as a resident worker.
#
# No network, no key material and no side effects live here.
set -eu

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec sleep infinity
