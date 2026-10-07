# Distributed Vector Processing Using MPI
## Lab Evaluation Presentation Deck — Topic No. 5

---

### SLIDE 1: Title Slide
* **Project Title:** Distributed Vector Processing Using Message Passing Interface (MPI)
* **Topic Number:** Topic No. 5
* **Course:** Parallel Computing (Mini-Project & Lab Evaluation)
* **Technology Stack:** Ubuntu / Linux, C (C99), MPICH / OpenMPI, GCC, Makefile, Python 3
* **Presenter:** [Your Name / Roll No / Team 5]

---

### SLIDE 2: Problem Statement & Motivation
* **Assigned Problem (PDF):**
  > *"Divide a large vector among processes and perform computations."*
* **Core Problem:**
  Sequential array processing is constrained by single-CPU clock frequency and memory bus bandwidth. For massive vectors ($10^7+$ elements), single-core processing creates execution bottlenecks.
* **Solution Approach:**
  Apply domain decomposition across multiple MPI processes, distribute data chunks via collective communications, process concurrently, and gather the final result back to the root process.

---

### SLIDE 3: Sequential vs. Distributed Memory Models
* **Shared Memory (e.g., OpenMP):**
  * Single address space; all threads share RAM.
  * Limited to a single computer motherboard (cannot scale to cluster).
* **Distributed Memory (MPI):**
  * Each process has its own **isolated private RAM**.
  * No shared variables; memory cannot be corrupted across processes.
  * Processes communicate strictly via **explicit network messages**.
  * Scales across multi-core processors, workstations, and high-performance supercomputers.

---

### SLIDE 4: Mathematical Formulation
* **Input Vector:** $X$ of size $N$ (e.g., $10^6, 5 \times 10^6, 10^7$ double-precision floats).
* **Element-Wise Transformation Function:**
  $$Y[i] = \sqrt{X[i]^2 + 2.5 \cdot X[i] + 1.0}$$
* **Global Reduction (Verification Checksum):**
  $$S = \sum_{i=0}^{N-1} Y[i]$$
* **Why this function?**
  It provides meaningful floating-point computational intensity (squaring, scaling, square root), exercising CPU arithmetic units and demonstrating measurable parallel speedup.

---

### SLIDE 5: Workload Partitioning & MPI Architecture
* **Partitioning Strategy:** 1D Block Decomposition.
* **Local Chunk Size:**
  $$n_{\text{local}} = \frac{N}{P}$$
* **Process Roles:**
  * **Process 0 (Root/Master):** Holds initial input array $X$ and final output array $Y$.
  * **Processes $0, \dots, P-1$ (Workers):** Allocate local chunk buffers $X_{\text{local}}$ and $Y_{\text{local}}$ of size $n_{\text{local}}$.

```text
Full Vector X [1 .. N] (Process 0)
        │
   MPI_Scatter()
        ├── Chunk 0 ──> Process 0 (local computation)
        ├── Chunk 1 ──> Process 1 (local computation)
        ├── Chunk 2 ──> Process 2 (local computation)
        └── Chunk 3 ──> Process 3 (local computation)
        │
   MPI_Gather()
        ▼
Full Vector Y [1 .. N] (Gathered back on Process 0)
```

---

### SLIDE 6: Key MPI Primitives Used
1. `MPI_Init(&argc, &argv)`: Initializes the MPI execution environment.
2. `MPI_Comm_size(MPI_COMM_WORLD, &P)`: Queries total number of participating processes.
3. `MPI_Comm_rank(MPI_COMM_WORLD, &rank)`: Queries unique integer ID of the caller ($0 \dots P-1$).
4. `MPI_Scatter(...)`: Distributes equal chunks of $X$ from Process 0 to all processes.
5. `MPI_Gather(...)`: Collects processed chunks from all processes back into array $Y$ on Process 0.
6. `MPI_Reduce(...)`: Sums all local partial sums into the global sum on Process 0 using `MPI_SUM`.
7. `MPI_Barrier(...)`: Synchronizes processes before starting/stopping wall clock timers.
8. `MPI_Wtime()`: High-resolution wall-clock timestamp generator.
9. `MPI_Finalize()`: Cleans up communication contexts and exits MPI safely.

---

### SLIDE 7: Correctness Verification
* **Requirement:** MPI results must exactly match sequential baseline results.
* **Automated Script:** `scripts/verify_correctness.sh`
* **Checks Performed:**
  1. Aggregate Checksum: $|S_{\text{seq}} - S_{\text{mpi}}| < 10^{-5}$
  2. Element-by-element diff: $\max |Y_{\text{seq}}[i] - Y_{\text{mpi}}[i]| < 10^{-6}$
* **Result:**
  ```text
  RESULT: PASS — sequential and MPI results match
  ```

---

### SLIDE 8: Experimental Setup & Benchmark Matrix
* **Platform:** Linux / Ubuntu (WSL2 environment).
* **Compiler:** GCC / MPICC with `-O3` optimization level.
* **Process Topologies ($P$):** 1, 2, 4, 8 processes.
* **Dataset Sizes ($N$):**
  * Small: $1,000,000$ elements ($10^6$)
  * Medium: $5,000,000$ elements ($5 \times 10^6$)
  * Large: $10,000,000$ elements ($10^7$)
* **Recorded Metrics:**
  * Sequential Time ($T_{\text{seq}}$)
  * Parallel Wall Time ($T_{\text{parallel}}$)
  * Pure Computation Time ($T_{\text{comp}}$)
  * Scatter/Gather Communication Time ($T_{\text{comm}}$)

---

### SLIDE 9: Performance Formulas
1. **Speedup ($S_P$):**
   $$S_P = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$$
   *Measures how many times faster parallel execution is compared to single-core execution.*

2. **Parallel Efficiency ($E_P$):**
   $$E_P = \frac{S_P}{P} = \frac{T_{\text{sequential}}}{P \times T_{\text{parallel}}}$$
   *Measures CPU utilization per core (ideal = 1.0 or 100%).*

---

### SLIDE 10: Performance Visualizations (Graphs)
*(Plots generated automatically into `graphs/` directory)*
1. **Execution Time vs. Processes:** Demonstrates consistent drop in wall-clock time as $P$ increases.
2. **Speedup vs. Processes:** Tracks sub-linear speedup curve against ideal linear line ($S = P$).
3. **Efficiency vs. Processes:** Illustrates efficiency drop due to communication overhead and Amdahl's Law serial fraction.

---

### SLIDE 11: Why Speedup is Not Perfectly Linear
1. **Communication Overhead:**
   Scattering input vector and gathering result vector takes finite network/IPC time ($T_{\text{comm}}$).
2. **Amdahl's Law:**
   Memory allocation, initialization, and root-level reductions are serial fractions ($f_{\text{serial}}$).
   $$\text{Max Speedup} \le \frac{1}{f + \frac{1-f}{P}}$$
3. **Hardware Resource Contention:**
   Processes running on the same machine share L3 cache and memory bus bandwidth.

---

### SLIDE 12: Checkpoints & Deliverables Checklist
* [x] **Checkpoint 1:** Problem definition + sequential algorithm + parallel design (2/2)
* [x] **Checkpoint 2:** Working parallel implementation in C with MPI collectives (2/2)
* [x] **Checkpoint 3:** Complete benchmarks with multiple data sizes & process counts (2/2)
* [x] **Checkpoint 4:** Graphs & analysis for time, speedup, and efficiency (2/2)
* [x] **Checkpoint 5:** Viva preparation & lab presentation ready (2/2)

---

### SLIDE 13: Conclusion
* Successfully implemented distributed vector processing for Topic No. 5 using MPI.
* Validated 100% mathematical accuracy against sequential baseline.
* Observed clear execution time reduction and speedup scaling.
* Demonstrates foundational concepts of distributed memory parallelism.
