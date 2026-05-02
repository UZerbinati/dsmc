#!/usr/bin/env bash
# Run all CFMZ needle (calamitic) DSMC tests in sequence.
#
# Covers test_needle_0 .. test_needle_28 (homogeneous needle solver) and
# test_needle_inhomo_0 (inhomogeneous needle solver, Sod-tube IC).
# Discotic tests live in run_disc.sh.
#
# Usage: ./run_needle.sh [-n <nprocs>] [-nlocal <nlocal>]
# Defaults: 10 MPI processes, 1 000 000 particles per rank.

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

# Numeric ordering rather than lexicographic so test_needle_2 runs before _10.
mapfile -t NEEDLE_TESTS < <(ls test_needle_*.py 2>/dev/null | sort -t_ -k3 -n)

for t in "${NEEDLE_TESTS[@]}"; do
    run_test "$t"
done

echo "=============================="
echo "Needle results: $PASS passed, $FAIL failed"
if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do echo "  $t"; done
    exit 1
fi
