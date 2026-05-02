#!/usr/bin/env bash
# Run all CFMZ discotic DSMC tests in sequence.
#
# Covers test_disc_0 .. test_disc_5 (homogeneous discotic solver,
# CFMZDiscDSMCHomo) and test_disc_6 (inhomogeneous discotic solver,
# CFMZDiscDSMC).  Calamitic / needle tests live in run_needle.sh.
#
# Usage: ./run_disc.sh [-n <nprocs>] [-nlocal <nlocal>]
# Defaults: 4 MPI processes, 250 000 particles per rank.

set -eo pipefail

NPROCS=4
NLOCAL=250000

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

# Numeric ordering on the trailing index.
mapfile -t DISC_TESTS < <(ls test_disc_*.py 2>/dev/null | sort -t_ -k3 -n)

for t in "${DISC_TESTS[@]}"; do
    run_test "$t"
done

echo "=============================="
echo "Disc results: $PASS passed, $FAIL failed"
if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do echo "  $t"; done
    exit 1
fi
