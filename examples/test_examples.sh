#!/usr/bin/env bash
set -eo pipefail

EXAMPLES_DIR=$(dirname "${BASH_SOURCE[0]}")

for example in ${EXAMPLES_DIR}/pipeline_*.py; do
    echo "Running example: ${example}"
    python ${example} 2>/dev/null
done