#!/usr/bin/env bash
# ==============================================================================
# Performance Benchmark & Experiment Runner
# File: scripts/run_experiments.sh
# Topic: 5 (Distributed Vector Processing)
#
# Runs sequential baseline and MPI parallel program across multiple
# problem sizes and process counts, computing execution time, speedup,
# and parallel efficiency.
# ==============================================================================

set -e

# Directory resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQ_BIN="${ROOT_DIR}/sequential"
MPI_BIN="${ROOT_DIR}/mpi_program"
RESULTS_DIR="${ROOT_DIR}/results"
CSV_FILE="${RESULTS_DIR}/timing_results.csv"

mkdir -p "${RESULTS_DIR}"

# Ensure binaries are built
if [ ! -f "${SEQ_BIN}" ] || [ ! -f "${MPI_BIN}" ]; then
    echo "Compiling binaries via Makefile..."
    make -C "${ROOT_DIR}" all
fi

# Initialize CSV file with clear headers
echo "Size_N,Processes,Seq_Time,Parallel_Time,Comp_Time,Comm_Time,Speedup,Efficiency" > "${CSV_FILE}"

# Test configurations
SIZES=(1000000 5000000 10000000)
PROCESSES=(1 2 4 8)

echo "=========================================================================="
echo "      DISTRIBUTED VECTOR PROCESSING: PERFORMANCE EXPERIMENTS              "
echo "=========================================================================="
printf "%-12s | %-6s | %-10s | %-10s | %-8s | %-10s\n" \
       "Input Size" "Procs" "Seq Time(s)" "Par Time(s)" "Speedup" "Efficiency"
echo "--------------------------------------------------------------------------"

for N in "${SIZES[@]}"; do
    # 1. Run Sequential Baseline
    SEQ_OUTPUT=$("${SEQ_BIN}" "${N}")
    SEQ_TIME=$(echo "${SEQ_OUTPUT}" | grep "Sequential Execution Time" | awk '{print $5}')
    
    # 2. Run Parallel MPI for each process count
    for P in "${PROCESSES[@]}"; do
        # Use --oversubscribe flag if OpenMPI supports/requires it for core counts > physical cores
        MPI_LINE=$(mpirun --oversubscribe -np "${P}" "${MPI_BIN}" "${N}" --csv 2>/dev/null || \
                   mpirun -np "${P}" "${MPI_BIN}" "${N}" --csv)
        
        # Parse CSV output from MPI program: ProcessCount,VectorSize,TotalTime,CompTime,CommTime,Checksum
        PAR_TIME=$(echo "${MPI_LINE}" | cut -d',' -f3)
        COMP_TIME=$(echo "${MPI_LINE}" | cut -d',' -f4)
        COMM_TIME=$(echo "${MPI_LINE}" | cut -d',' -f5)

        # Calculate Speedup and Efficiency using Python for high floating-point precision
        METRICS=$(python3 - <<EOF
seq_t = float("${SEQ_TIME}")
par_t = float("${PAR_TIME}")
p = int("${P}")

speedup = seq_t / par_t if par_t > 0 else 0.0
efficiency = speedup / p if p > 0 else 0.0

print(f"{speedup:.4f},{efficiency:.4f}")
EOF
        )
        SPEEDUP=$(echo "${METRICS}" | cut -d',' -f1)
        EFFICIENCY=$(echo "${METRICS}" | cut -d',' -f2)

        # Append to CSV
        echo "${N},${P},${SEQ_TIME},${PAR_TIME},${COMP_TIME},${COMM_TIME},${SPEEDUP},${EFFICIENCY}" >> "${CSV_FILE}"

        # Print formatted row
        printf "%-12s | %-6d | %-11.4f | %-11.4f | %-8.2f | %-10.2f\n" \
               "${N}" "${P}" "${SEQ_TIME}" "${PAR_TIME}" "${SPEEDUP}" "${EFFICIENCY}"
    done
    echo "--------------------------------------------------------------------------"
done

echo "Benchmark finished successfully!"
echo "Raw timing results saved to: ${CSV_FILE}"
