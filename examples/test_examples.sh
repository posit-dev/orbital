#!/usr/bin/env bash
EXAMPLES_DIR=$(dirname "${BASH_SOURCE[0]}")

export PRINT_SQL=1 # Test that SQL generation runs.
export ASSERT=1  # Enable assertions in the examples

for example in ${EXAMPLES_DIR}/pipeline_*.py; do
    echo "Running example: ${example}"
    time python ${example} > test_examples.log 2>&1
    if [ $? -ne 0 ]; then
        echo "Error running example: ${example}"
        cat test_examples.log
        exit 1
    fi
    tail -n 1 test_examples.log
done
