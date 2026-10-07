# Distributed Vector Processing Using Message Passing Interface (MPI)
## Parallel Computing Mini-Project Report — Topic No. 5

---

### Abstract
This project implements and evaluates **Distributed Vector Processing** under the **Message Passing Interface (MPI)** paradigm, fulfilling all specifications outlined in **Topic No. 5** of the Parallel Computing Mini-Project. Large-scale numerical array transformations are fundamental to scientific simulations, image processing, and numerical physics. Under a sequential single-threaded execution model, vector operations are bound by both single-core execution throughput and memory bus bandwidth. In this project, a distributed-memory parallel processing pipeline is designed and implemented in C using standard MPI collective operations (`MPI_Scatter`, `MPI_Gather`, and `MPI_Reduce`). Experiments are conducted across multiple problem sizes ($N = 10^6, 5 \times 10^6, 10^7$) and process topologies ($P = 1, 2, 4, 8$). Correctness is rigorously validated against a sequential baseline, and performance is quantified using execution time, parallel speedup ($S_p$), and parallel efficiency ($E_p$).

---

### 1. Introduction and Theoretical Background

#### 1.1 Parallel Computing Paradigms
Parallel computing involves the simultaneous execution of computational tasks across multiple processing elements to reduce total execution time. Modern parallel systems predominantly follow two architectural memory models:
1. **Shared Memory Architecture (e.g., OpenMP):** All processing units (threads) access a single global address space. While inter-thread communication is fast, scalability is physically constrained by bus bandwidth and cache coherency protocols on a single motherboard.
2. **Distributed Memory Architecture (e.g., MPI):** Each process possesses its own private, isolated memory address space. No process can directly read or write to another process's RAM. Processes coordinate strictly by exchanging explicit network messages. This enables horizontal scaling across arbitrary numbers of nodes in supercomputing clusters.

#### 1.2 The Message Passing Interface (MPI)
MPI is the de-facto standard specification for distributed memory programming. It defines point-to-point and collective communication primitives that abstract low-level network networking protocols (TCP/IP, InfiniBand, shared memory IPC).

---

### 2. Problem Statement & Mathematical Formulation

According to **Topic No. 5**, the core objective is:
> *"Divide a large vector among processes and perform computations."*

#### 2.1 Mathematical Transformation Kernel
Given an input vector $X \in \mathbb{R}^N$, each element undergoes an arithmetic transformation:
$$Y[i] = f(X[i]) = \sqrt{X[i]^2 + 2.5 \cdot X[i] + 1.0}, \quad \forall i \in \{0, 1, \dots, N-1\}$$

In addition, an aggregate reduction (global sum) is computed:
$$S = \sum_{i=0}^{N-1} Y[i]$$

This kernel exhibits high floating-point arithmetic intensity (squaring, scaling, square-root evaluation), avoiding artificial memory bus saturation and demonstrating real parallel compute speedup.

---

### 3. Algorithm Design and Architecture

#### 3.1 Domain Decomposition (1D Block Partitioning)
The global vector of size $N$ is divided equally among $P$ MPI processes:
$$\text{Local Chunk Size } n_{\text{local}} = \frac{N}{P}$$
* Process $p$ is responsible for elements with global indices:
  $$\text{Global Indices for Process } p = [p \cdot n_{\text{local}}, \; (p + 1) \cdot n_{\text{local}} - 1]$$

#### 3.2 MPI Collective Operations Workflow
```text
               Process 0 (Root Master)
            [ Full Vector X (Size N) ]
                         │
                 MPI_Scatter()
        ┌───────────┬────┴──────┬───────────┐
        ▼           ▼           ▼           ▼
    Process 0   Process 1   Process 2   Process P-1
    [X_loc_0]   [X_loc_1]   [X_loc_2]   [X_loc_P-1]
        │           │           │           │
     Compute     Compute     Compute     Compute
    [Y_loc_0]   [Y_loc_1]   [Y_loc_2]   [Y_loc_P-1]
        │           │           │           │
        └───────────┼────┬──────┴───────────┘
                 MPI_Gather()
                         ▼
               Process 0 (Root Master)
            [ Full Vector Y (Size N) ]
```

1. **Initialization:** `MPI_Init` boots the MPI runtime; `MPI_Comm_size` queries $P$; `MPI_Comm_rank` queries process ID $rank \in [0, P-1]$.
2. **Buffer Allocation:**
   * Process 0 allocates global vectors $X$ and $Y$ of size $N$.
   * Every process allocates private local vectors $X_{\text{local}}$ and $Y_{\text{local}}$ of size $n_{\text{local}}$.
3. **Data Distribution (`MPI_Scatter`):**
   Process 0 splits $X$ into chunks of $n_{\text{local}}$ double-precision floats and distributes one chunk to each process.
4. **Local Computation:**
   Each process independently evaluates $Y_{\text{local}}[j] = f(X_{\text{local}}[j])$ in its private RAM.
5. **Data Collection (`MPI_Gather`):**
   Process 0 collects all computed $Y_{\text{local}}$ chunks into the global array $Y$.
6. **Global Sum Reduction (`MPI_Reduce`):**
   Process local partial sums are combined into $S_{\text{global}}$ on Process 0 using the `MPI_SUM` operator.
7. **Finalization:** `MPI_Finalize` cleanly tears down MPI communication channels.

---

### 4. Correctness Verification Methodology

A parallel program is useless if its calculations differ from mathematical truth. We employ an automated verification script (`scripts/verify_correctness.sh`) that compares:
1. **Aggregate Checksum:** $|S_{\text{seq}} - S_{\text{mpi}}| < 10^{-5}$
2. **Element-wise Integrity:** $\max_{i} |Y_{\text{seq}}[i] - Y_{\text{mpi}}[i]| < 10^{-6}$

If both conditions hold, the system outputs:
```text
RESULT: PASS — sequential and MPI results match
```

---

### 5. Performance Evaluation Metrics

1. **Execution Time ($T_P$):** Total wall-clock time elapsed from starting the distribution to collecting the final gathered results, measured via `MPI_Wtime()`.
2. **Speedup ($S_P$):**
   $$S_P = \frac{T_{\text{sequential}}}{T_{\text{parallel}}(P)}$$
3. **Parallel Efficiency ($E_P$):**
   $$E_P = \frac{S_P}{P} = \frac{T_{\text{sequential}}}{P \times T_{\text{parallel}}(P)}$$
4. **Communication Overhead Ratio:**
   $$\text{Overhead Ratio} = \frac{T_{\text{comm}}}{T_{\text{parallel}}}$$

---

### 6. Summary of Academic Checkpoints

| Checkpoint | Description | Status |
| :--- | :--- | :--- |
| **Checkpoint 1** | Problem definition + sequential algorithm + parallel design | Completed |
| **Checkpoint 2** | Working parallel implementation using assigned MPI model | Completed |
| **Checkpoint 3** | Run with multiple data sizes and process counts (1, 2, 4, 8) | Completed |
| **Checkpoint 4** | Generate plots and analyze execution time, speedup, efficiency | Completed |
| **Checkpoint 5** | Technical demonstration and viva preparation | Completed |

---

### 7. Conclusion
Distributed vector processing using MPI achieves substantial execution time reductions for large array transformations. Collective communication routines (`MPI_Scatter` and `MPI_Gather`) offer clean abstractions for domain partitioning. As data sizes scale, computation time increasingly dominates communication overhead, yielding high parallel efficiency.
