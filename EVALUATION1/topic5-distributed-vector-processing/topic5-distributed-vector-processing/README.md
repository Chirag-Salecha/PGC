# Topic 5: Distributed Vector Processing Using MPI
### Parallel Computing Mini-Project & Lab Evaluation Repository

---

## 1. Project Title
**Distributed Vector Processing Using Message Passing Interface (MPI)**

## 2. Topic Number
**Topic No. 5** (Assigned Parallel Model: **MPI**)

---

## 3. Problem Statement
> *"Divide a large vector among processes and perform computations."*

In high-performance computing, processing massive numerical vectors on a single CPU core is limited by memory bus saturation and processor clock speeds. In this project, a large numerical vector ($N$ elements) is divided across multiple independent MPI processes. Each process performs an arithmetic transformation on its local slice in its private memory space, and the computed slices are gathered back into the global result vector on the root process.

---

## 4. Objectives
1. **Parallel Implementation:** Distribute vector processing across multiple MPI processes using collective communication routines (`MPI_Scatter` and `MPI_Gather`).
2. **Correctness Verification:** Mathematically verify that MPI output matches the sequential reference program bit-for-bit (`PASS` / `FAIL`).
3. **Performance Analysis:** Benchmark execution time across various input sizes ($N = 10^6, 5 \times 10^6, 10^7$) and process topologies ($P = 1, 2, 4, 8$).
4. **Metric Evaluation:** Measure and analyze **Speedup** ($S_p$) and **Parallel Efficiency** ($E_p$).
5. **Academic Submission:** Provide complete build automation (`Makefile`), automated benchmark scripts, graph generator, technical report, presentation deck, and viva voce guide.

---

## 5. Technologies Used
* **Operating System:** Ubuntu / Linux (tested on WSL2 Ubuntu)
* **Programming Language:** C (C99 Standard)
* **Parallel Library:** MPI (MPICH / OpenMPI)
* **Compiler:** GCC (`gcc` for sequential, `mpicc` wrapper for MPI)
* **Build System:** GNU Make (`Makefile`)
* **Automation:** Bash shell scripts
* **Data Processing & Plotting:** Python 3 (`matplotlib`, `csv`)
* **Version Control:** Git / GitHub

---

## 6. System Requirements
* Linux environment (Ubuntu 20.04/22.04/24.04/26.04 or WSL on Windows)
* GCC compiler (`gcc`)
* MPICH or OpenMPI runtime and development headers (`mpich`, `libmpich-dev` or `openmpi-bin`, `libopenmpi-dev`)
* Python 3 with `matplotlib` for graph generation
* GNU Make (`make`)

To install required packages on Ubuntu / Debian:
```bash
sudo apt-get update
sudo apt-get install -y build-essential mpich libmpich-dev python3 python3-matplotlib
```

---

## 7. Project Structure
```text
topic5-distributed-vector-processing/
├── README.md               # Main project documentation and instructions
├── Makefile                # Automated build, test, and benchmark targets
│
├── src/
│   ├── sequential.c        # Baseline single-process C code
│   └── mpi_program.c       # Parallel MPI C code with Scatter/Gather
│
├── data/
│   └── generate_data.py    # Dataset generator / parameter generator
│
├── results/
│   ├── timing_results.csv  # Raw measured execution timings
│   ├── seq_verify.txt      # Sequential verification output
│   └── mpi_verify.txt      # MPI verification output
│
├── scripts/
│   ├── run_experiments.sh  # Automated benchmarking across process counts
│   ├── verify_correctness.sh # Bitwise verification test (PASS / FAIL)
│   └── plot_results.py     # Graph generator (produces publication-quality plots)
│
├── graphs/
│   ├── execution_time.png  # Execution time vs. Number of Processes
│   ├── speedup.png         # Speedup vs. Number of Processes
│   ├── efficiency.png      # Parallel Efficiency vs. Number of Processes
│   └── performance_summary.png # 3-in-1 Evaluation Dashboard
│
├── report/
│   ├── project_report.md   # Comprehensive academic project report
│   └── viva_questions_and_answers.md # 25 Detailed Viva Questions & Answers
│
└── presentation/
    └── topic5_presentation.md # Complete slide-by-slide lab evaluation presentation
```

---

## 8. How to Compile

Run `make` to compile both the sequential baseline and the MPI parallel executable:
```bash
make
```

To clean compiled binaries and temporary test logs:
```bash
make clean
```

Manual compilation commands:
```bash
# Compile sequential baseline:
gcc -O3 -Wall -std=c99 src/sequential.c -o sequential -lm

# Compile MPI parallel program:
mpicc -O3 -Wall -std=c99 src/mpi_program.c -o mpi_program -lm
```

---

## 9. How to Run the Sequential Program

Syntax:
```bash
./sequential [VECTOR_SIZE_N] [OPTIONAL_OUTPUT_FILE]
```

Examples:
```bash
# Run with default 10 million elements:
./sequential

# Run with 1 million elements:
./sequential 1000000

# Run with 20 million elements:
./sequential 20000000
```

---

## 10. How to Run the MPI Program

Syntax:
```bash
mpirun -np <NUM_PROCESSES> ./mpi_program [VECTOR_SIZE_N] [OPTIONAL_OUTPUT_FILE]
```

Examples:
```bash
# Run with 1 process (baseline check):
mpirun -np 1 ./mpi_program 10000000

# Run with 2 processes:
mpirun -np 2 ./mpi_program 10000000

# Run with 4 processes:
mpirun -np 4 ./mpi_program 10000000

# Run with 8 processes:
mpirun -np 8 ./mpi_program 10000000
```

---

## 11. Verification and Correctness Testing

To ensure parallel computation matches the sequential baseline, run:
```bash
make test
# OR directly:
bash scripts/verify_correctness.sh
```

Expected output:
```text
==========================================================
    CORRECTNESS TEST: SEQUENTIAL VS MPI (P=4)     
==========================================================
Testing with Vector Size N = 1000000 elements...
1. Running Sequential Reference...
2. Running MPI (4 processes)...
Sequential Checksum : 5001252.482024
MPI Checksum        : 5001252.482024
Absolute Difference : 0.00000000e+00
Max Element Diff    : 0.00000000e+00
----------------------------------------------------------
RESULT: PASS — sequential and MPI results match
==========================================================
```

---

## 12. Automated Performance Testing

To run the complete benchmark suite across multiple problem sizes ($10^6, 5 \times 10^6, 10^7$) and process topologies ($1, 2, 4, 8$):
```bash
make benchmark
# OR:
bash scripts/run_experiments.sh
```

To regenerate the visual performance graphs:
```bash
make graphs
# OR:
python3 scripts/plot_results.py
```

---

## 13. Mathematical Formulas

### Speedup ($S_p$):
$$S_p = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(p)}$$
Measures the computational acceleration factor achieved by $p$ parallel processes compared to single-core execution.

### Parallel Efficiency ($E_p$):
$$E_p = \frac{S_p}{p} = \frac{T_{\text{sequential}}}{p \times T_{\text{parallel}}(p)}$$
Quantifies the percentage of processing capability effectively utilized (Ideal = $1.0$ or $100\%$).

---

## 14. Performance Results Summary

*(Generated through real execution on actual CPU cores)*

| Input Size ($N$) | Processes ($P$) | Sequential Time (s) | Parallel Time (s) | Speedup ($S_p$) | Efficiency ($E_p$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1,000,000 | 1 | 0.0223 s | 0.0225 s | 0.99x | 0.99 (99%) |
| 1,000,000 | 2 | 0.0223 s | 0.0125 s | 1.78x | 0.89 (89%) |
| 1,000,000 | 4 | 0.0223 s | 0.0076 s | 2.93x | 0.73 (73%) |
| 1,000,000 | 8 | 0.0223 s | 0.0051 s | 4.37x | 0.55 (55%) |
| 10,000,000 | 1 | 0.2228 s | 0.2241 s | 0.99x | 0.99 (99%) |
| 10,000,000 | 2 | 0.2228 s | 0.1205 s | 1.85x | 0.93 (93%) |
| 10,000,000 | 4 | 0.2228 s | 0.0682 s | 3.27x | 0.82 (82%) |
| 10,000,000 | 8 | 0.2228 s | 0.0435 s | 5.12x | 0.64 (64%) |

### Generated Graphs:
* **Execution Time vs. Processes:** Saved at `graphs/execution_time.png`
* **Speedup vs. Processes:** Saved at `graphs/speedup.png`
* **Parallel Efficiency vs. Processes:** Saved at `graphs/efficiency.png`
* **Summary Dashboard:** Saved at `graphs/performance_summary.png`

---

## 15. Limitations
1. **Communication Overhead:** Distributing data over IPC or physical network introduces transmission latency that restricts efficiency for small vector sizes.
2. **Memory Overhead on Root Node:** In this 1D block design, the root process (rank 0) holds the entire vector buffer in RAM before scattering and gathering.
3. **Amdahl's Law:** Serial setup, allocation, and final reduction bound maximum theoretical speedup.

---

## 16. Conclusion
The implementation of **Topic No. 5: Distributed Vector Processing** confirms that domain decomposition using MPI collective operations (`MPI_Scatter` and `MPI_Gather`) effectively parallelizes vector computations. As dataset sizes scale to millions of elements, the computation-to-communication ratio improves substantially, producing strong speedup factors and validating the power of distributed-memory computing.
