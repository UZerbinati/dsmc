#!/usr/bin/env bash
# Run all CFM DSMC tests in sequence.
# Usage: ./run_tests.sh [-n <nprocs>] [-nlocal <nlocal>]
#
# Defaults: 5 MPI processes, 10 000 000 particles per rank.

set -eo pipefail

NPROCS=10
NLOCAL=1000000

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)      NPROCS="$2";  shift 2 ;;
        -nlocal) NLOCAL="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
FAILED_TESTS=()

run_test() {
    local script="$1"
    echo "--- $script ---"
    if mpirun --use-hwthread-cpus -n "$NPROCS" python "$script" -nlocal "$NLOCAL" 2>&1; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$script")
    fi
    echo
}

for t in test_*.py; do
    run_test "$t"
done

echo "=============================="
echo "Results: $PASS passed, $FAIL failed"
if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do echo "  $t"; done
    exit 1
fi
