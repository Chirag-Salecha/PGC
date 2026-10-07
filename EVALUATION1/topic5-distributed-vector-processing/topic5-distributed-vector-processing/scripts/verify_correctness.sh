#!/usr/bin/env bash
# ==============================================================================
# Correctness Verification Script
# File: scripts/verify_correctness.sh
# Topic: 5 (Distributed Vector Processing)
#
# Compares the output of the sequential baseline and the MPI parallel program
# to mathematically verify that the distributed computation is 100% correct.
# ==============================================================================

set -e

# Directory resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQ_BIN="${ROOT_DIR}/sequential"
MPI_BIN="${ROOT_DIR}/mpi_program"
RESULTS_DIR="${ROOT_DIR}/results"

mkdir -p "${RESULTS_DIR}"

TEST_N=1000000
PROCS=4

echo "=========================================================="
echo "    CORRECTNESS TEST: SEQUENTIAL VS MPI (P=${PROCS})     "
echo "=========================================================="
echo "Testing with Vector Size N = ${TEST_N} elements..."

SEQ_DUMP="${RESULTS_DIR}/seq_verify.txt"
MPI_DUMP="${RESULTS_DIR}/mpi_verify.txt"

# 1. Run sequential
echo "1. Running Sequential Reference..."
"${SEQ_BIN}" "${TEST_N}" "${SEQ_DUMP}" > /dev/null

# 2. Run MPI
echo "2. Running MPI (${PROCS} processes)..."
mpirun -np "${PROCS}" "${MPI_BIN}" "${TEST_N}" "${MPI_DUMP}" > /dev/null

# 3. Compare results using Python validation logic
python3 - <<EOF
import sys

def read_verify_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    chksum = float(lines[1])
    time_taken = float(lines[2])
    elements = [float(x) for x in lines[3:]]
    return n, chksum, time_taken, elements

seq_n, seq_chksum, seq_t, seq_elems = read_verify_file("${SEQ_DUMP}")
mpi_n, mpi_chksum, mpi_t, mpi_elems = read_verify_file("${MPI_DUMP}")

diff_chksum = abs(seq_chksum - mpi_chksum)
rel_diff = diff_chksum / abs(seq_chksum) if abs(seq_chksum) > 0 else 0

max_elem_diff = 0.0
for s, m in zip(seq_elems, mpi_elems):
    diff = abs(s - m)
    if diff > max_elem_diff:
        max_elem_diff = diff

print(f"Sequential Checksum : {seq_chksum:.6f}")
print(f"MPI Checksum        : {mpi_chksum:.6f}")
print(f"Absolute Difference : {diff_chksum:.8e}")
print(f"Max Element Diff    : {max_elem_diff:.8e}")
print("----------------------------------------------------------")

TOLERANCE = 1e-5
if rel_diff < TOLERANCE and max_elem_diff < TOLERANCE:
    print("RESULT: PASS — sequential and MPI results match")
    sys.exit(0)
else:
    print("RESULT: FAIL — results do not match")
    sys.exit(1)
EOF

echo "=========================================================="
