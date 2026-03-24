#!/usr/bin/env bash
# Run all DSMC tests in parallel.
# Usage: ./run_tests.sh [-n <nprocs>] [-nlocal <nlocal>] [-nsteps <nsteps>]
#
# Defaults: 2 MPI processes, 500 particles per rank, 5 steps.

set -e

NPROCS=2
NLOCAL=500
NSTEPS=5

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)      NPROCS="$2";  shift 2 ;;
        -nlocal) NLOCAL="$2";  shift 2 ;;
        -nsteps) NSTEPS="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PASS=0
FAIL=0
FAILED_TESTS=()

run_test() {
    local script="$1"
    echo "--- $script ---"
    if mpirun -n "$NPROCS" python "$script" -nlocal "$NLOCAL" -nsteps "$NSTEPS" 2>&1 | tail -3; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_TESTS+=("$script")
    fi
    echo
}

for t in cfmz/test_*.py; do
    run_test "$t"
done

for t in boltzmann/test_*.py; do
    run_test "$t"
done

echo "=============================="
echo "Results: $PASS passed, $FAIL failed"
if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do echo "  $t"; done
    exit 1
fi
