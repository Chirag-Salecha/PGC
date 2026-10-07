# Multithreaded Programming in C — Pthreads vs. OpenMP

> **Parallel & GPU Computing Lab — Experiment 2**  
> *Comparative Study of Low-Level POSIX Threads and Directive-Based OpenMP Concurrency*

[![Language: C](https://img.shields.io/badge/Language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Compiler: GCC](https://img.shields.io/badge/Compiler-GCC-orange.svg)](https://gcc.gnu.org/)
[![API: POSIX Threads](https://img.shields.io/badge/Threading-Pthreads-green.svg)](https://en.wikipedia.org/wiki/Pthreads)
[![API: OpenMP](https://img.shields.io/badge/Parallel-OpenMP-yellow.svg)](https://www.openmp.org/)
[![Charts: Matplotlib](https://img.shields.io/badge/Visuals-Matplotlib-9cf.svg)](https://matplotlib.org/)

---

## Executive Summary

This repository presents a systematic study and comparative benchmark of multithreaded concurrent programming in C using **POSIX Threads (Pthreads)** and **OpenMP**. The experiments systematically explore thread lifecycles, work partitioning, shared-memory concurrency, data race conditions, synchronization primitives (mutexes vs. critical sections), barrier coordination, and compiler reductions.

To quantitatively evaluate performance scalability, identical compute-heavy workloads ($N = 1,000,000,000$ iterations computing $\sum_{i=0}^{N-1} i \times 10^{-6}$) were measured across varying thread counts (**1, 2, 4, 6, and 16 worker threads**).

```mermaid
flowchart TD
    A["Compute Workload: N = 1,000,000,000 Iterations\nTarget Sum = 499999999500.00"] --> B["Execution Approaches"]
    
    B --> C["1. Sequential Baseline\nSingle CPU Thread\nTime: 3.346s"]
    B --> D["2. POSIX Threads (Pthreads)\nExplicit Thread Management\nManual Work Division & Joins"]
    B --> E["3. OpenMP Runtime\nDirective-Based Parallelism\nCompiler Work-Sharing & Reductions"]
    
    D --> F["Benchmark Across Threads\n1, 2, 4, 6, 16 Threads"]
    E --> F
    
    F --> G["Performance Metrics Analysis\n• Execution Time (s)\n• Speedup Factor (x)\n• Parallel Efficiency (%)"]
```

### Key Performance Findings
* **Sequential Baseline:** Single-threaded execution took **`3.346414 s`**.
* **Pthreads at 16 Threads:** Reduced runtime to **`0.530415 s`** (**`6.309x`** speedup, **`39.43%`** efficiency).
* **OpenMP at 16 Threads:** Reduced runtime to **`0.448971 s`** (**`7.454x`** speedup, **`46.58%`** efficiency).
* **Takeaway:** OpenMP outperformed manual Pthreads partitioning at high thread counts due to low-overhead runtime thread pooling and compiler-vectorized tree reductions, while both implementations clearly demonstrated Amdahl's Law and memory bus saturation as thread counts scaled to 16.

---

## Conceptual Overview: Pthreads vs. OpenMP

| Feature | POSIX Threads (Pthreads) | OpenMP |
|---|---|---|
| **Programming Paradigm** | Explicit library API (Low-level) | Compiler directives & runtime (High-level) |
| **Thread Creation** | Explicit call to `pthread_create()` | Implicit via `#pragma omp parallel` |
| **Work Partitioning** | Manual index boundary calculation by developer | Automatic scheduling via `#pragma omp parallel for` |
| **Synchronization** | Explicit mutex lock/unlock (`pthread_mutex_t`) | Structured blocks (`#pragma omp critical`) |
| **Coordination** | Explicit `pthread_join()` / condition variables | Implicit barrier at region end or `#pragma omp barrier` |
| **Reduction** | Manual partial sum buffers and serial aggregation | Built-in clause `reduction(+:variable)` |
| **Control Level** | Fine-grained, hardware-near thread control | Rapid development, directive-driven optimization |

### Race Conditions & Synchronization Mechanics
When multiple threads update a shared variable (`counter++`) concurrently without protection, a **race condition** occurs because `counter++` is not atomic at the CPU machine-instruction level:
1. **Read:** Register loads current memory value.
2. **Modify:** ALU increments the register value.
3. **Write:** Register stores updated value back to RAM.

Without synchronization, simultaneous reads cause interleaved writes to overwrite each other, causing lost updates.

```
Thread A:  [Read: 100] -----> [Increment: 101] -----> [Write: 101]
Thread B:       [Read: 100] -----> [Increment: 101] -------------> [Write: 101] (Update Lost!)
```

* **Pthreads Fix:** Mutual exclusion lock (`pthread_mutex_lock` / `pthread_mutex_unlock`) ensures only one thread executes the critical section at any moment.
* **OpenMP Fix:** Directive `#pragma omp critical` serializes execution of the enclosed block across all active threads.
* **Reduction Optimization:** Instead of serializing every addition with locks (which causes massive lock contention), each thread maintains a thread-private accumulator and combines results into a global accumulator at the end using `#pragma omp parallel for reduction(+:sum)`.

---

## Quantitative Benchmark Results

The benchmark calculates $\sum_{i=0}^{N-1} i \times 10^{-6}$ for $N = 10^9$ iterations. The verified correct output across all runs is:
$$\text{Result} = 499999999500.00$$

### Performance Metrics Table

| Worker Threads | Pthreads Time (s) | OpenMP Time (s) | Pthreads Speedup | OpenMP Speedup | Pthreads Efficiency | OpenMP Efficiency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline (1)** | 3.346414 | 3.346414 | 1.000x | 1.000x | 100.00% | 100.00% |
| **1 Thread** | 3.205682 | 3.186103 | 1.044x | 1.050x | 104.39% | 105.03% |
| **2 Threads** | 1.689293 | 1.662809 | 1.981x | 2.013x | 99.05% | 100.63% |
| **4 Threads** | 0.947512 | 0.978149 | 3.532x | 3.421x | 88.29% | 85.53% |
| **6 Threads** | 0.850071 | 0.841213 | 3.937x | 3.978x | 65.61% | 66.30% |
| **16 Threads** | **0.530415** | **0.448971** | **6.309x** | **7.454x** | **39.43%** | **46.58%** |

*Formulas:*
* $\text{Speedup } (S) = \frac{T_{\text{sequential}}}{T_{\text{parallel}}}$
* $\text{Efficiency } (E) = \frac{S}{p} \times 100\%$ *(where $p$ is the number of threads)*

---

## High-Resolution Performance Visualizations

### 1. Executive Performance Dashboard
Comprehensive 3-panel evaluation comparing execution duration, speedup factors, and parallel efficiency side-by-side:

![Performance Comparison Charts](images/performance_comparison_charts.png)

---

### 2. Execution Time Comparison (Lower is Better)
Execution duration drops rapidly as thread count increases from 1 to 4 threads, with OpenMP achieving the fastest turnaround of **`0.45s`** at 16 threads:

![Execution Time Chart](images/execution_time_chart.png)

---

### 3. Parallel Speedup Analysis (Higher is Better)
Speedup tracks near-linear scaling up to 4 threads, after which overhead dampens further gains:

![Speedup Chart](images/speedup_chart.png)

---

### 4. Parallel Efficiency Analysis
Demonstrates hardware scaling limits: efficiency remains near ~90–100% up to 4 cores, tapering to ~40–47% at 16 threads due to synchronization, cache contention, and memory bandwidth bounds:

![Efficiency Chart](images/efficiency_chart.png)

---

## Experimental Proof & Output Gallery

### Part A — POSIX Threads (Pthreads)

| Program | Objective | Verification Output Screenshot |
|---|---|---|
| `pthreads/thread1.c` | Basic thread creation and termination using `pthread_create()` and `pthread_join()` | ![Thread 1 Result](images/thread1_result.png) |
| `pthreads/multiple_threads.c` | Spawning and coordinating multiple independent worker threads | ![Multiple Threads Result](images/multiple_threads_result.png) |
| `pthreads/thread_sum.c` | Manual block-based workload distribution and partial summation | ![Thread Sum Result](images/thread_sum_result.png) |
| `pthreads/race.c` | Deliberate race condition demonstration with concurrent shared counter increments | ![Race Condition Result](images/racecondition_result.png) |
| `pthreads/mutex.c` | Eliminating data race using POSIX mutual exclusion (`pthread_mutex_t`) | ![Mutex Result](images/mutex_result.png) |
| `pthreads/pthread_perf.c` | Scaling benchmark measuring timing across 1, 2, 4, 6, and 16 worker threads | ![Pthreads Perf Result](images/pthread_perf_result.png) |

---

### Part B — OpenMP (Open Multi-Processing)

| Program | Objective | Verification Output Screenshot |
|---|---|---|
| `openmp/omp1.c` | Basic parallel region instantiation and thread ID reporting (`omp_get_thread_num`) | ![OpenMP 1 Result](images/omp1_result.png) |
| `openmp/omp_sum.c` | Work-sharing loop parallelization using directive reduction (`reduction(+:sum)`) | ![OpenMP Sum Result](images/omp_sum_result.png) |
| `openmp/omp_critical.c` | Eliminating concurrent modification race condition using `#pragma omp critical` | ![OpenMP Critical Result](images/omp_critical_result.png) |
| `openmp/omp_barrier.c` | Explicit thread synchronization and rendezvous points using `#pragma omp barrier` | ![OpenMP Barrier Result](images/omp_barrier_result.png) |
| `openmp/omp_perf.c` | Scaling benchmark measuring timing across 1, 2, 4, 6, and 16 OpenMP threads | ![OpenMP Perf Result](images/omp_perf_result.png) |

---

### Part C — Sequential Baseline

| Program | Objective | Verification Output Screenshot |
|---|---|---|
| `performance_analysis/sequential.c` | Single-core baseline execution timing for identical $10^9$ iteration workload (**3.346s**) | ![Sequential Result](images/sequential_result.png) |

---

## Repository Structure

```
PGC_LAB02/
├── images/                                 # High-resolution benchmark charts & terminal screenshots
│   ├── efficiency_chart.png
│   ├── execution_time_chart.png
│   ├── performance_comparison_charts.png
│   ├── speedup_chart.png
│   ├── multiple_threads_result.png
│   ├── mutex_result.png
│   ├── omp1_result.png
│   ├── omp_barrier_result.png
│   ├── omp_critical_result.png
│   ├── omp_perf_result.png
│   ├── omp_sum_result.png
│   ├── pthread_perf_result.png
│   ├── racecondition_result.png
│   ├── sequential_result.png
│   ├── thread1_result.png
│   └── thread_sum_result.png
│
├── openmp/                                 # OpenMP directive-based implementations
│   ├── omp1.c
│   ├── omp_barrier.c
│   ├── omp_critical.c
│   ├── omp_race.c
│   └── omp_sum.c
│
├── performance_analysis/                   # Performance benchmarking programs
│   ├── omp_perf.c
│   ├── pthread_perf.c
│   └── sequential.c
│
├── pthreads/                               # POSIX threads implementations
│   ├── multiple_threads.c
│   ├── mutex.c
│   ├── race.c
│   ├── thread1.c
│   └── thread_sum.c
│
├── scripts/                                # Utility and visualization scripts
│   └── generate_charts.py                  # Generates publication-ready performance charts
│
├── .gitignore                              # Git ignore rules for binaries and system files
└── README.md                               # Project documentation & benchmark report
```

---

## How to Clone and Run

To clone and run this experiment locally:

```bash
# 1. Clone the repository
git clone https://github.com/DivyaKumari29/PGC_LAB02.git

# 2. Enter repository directory
cd PGC_LAB02

# 3. Run Sequential Baseline
gcc -O2 performance_analysis/sequential.c -o sequential
./sequential

# 4. Run Pthreads Benchmark
gcc -pthread -O2 performance_analysis/pthread_perf.c -o pthread_perf
./pthread_perf

# 5. Run OpenMP Benchmark
gcc -fopenmp -O2 performance_analysis/omp_perf.c -o omp_perf
./omp_perf

# 6. Redraw the charts
python scripts/generate_charts.py
```

---

## How to Upload to GitHub

1. Go to **github.com** → click **+** → **New repository**.
2. Give it a name (`PGC_LAB02`) → click **Create repository**.
3. Click **Add file** → **Upload files**.
4. Drag in `README.md`, `.gitignore`, and the `images`, `openmp`, `performance_analysis`, `pthreads`, `scripts` folders.
5. Click **Commit changes**.

Or using Git commands:

```bash
git init
git add .
git commit -m "Add PGC Lab 02 Pthreads vs OpenMP experiment"
git branch -M main
git remote add origin https://github.com/DivyaKumari29/PGC_LAB02.git
git push -u origin main
```

---

## Key Conclusions

1. **Scalability:** Both Pthreads and OpenMP provide dramatic performance improvements over single-threaded sequential code, slashing execution time from **3.35s down to ~0.45s–0.53s** (a **>6x–7.4x** speedup).
2. **OpenMP Efficiency at Scale:** OpenMP achieved a slightly higher speedup at 16 threads (**7.45x** vs. **6.31x**) due to its highly optimized work-sharing thread pool and compiler-level reduction mechanisms.
3. **Diminishing Returns (Amdahl's Law):** Moving from 1 to 4 threads yields near-linear speedup (~88% efficiency), but scaling from 6 to 16 threads sees efficiency decline to ~40–47% as synchronization, thread orchestration, and hardware memory bus saturation begin to dominate runtime.
4. **Synchronization Tradeoffs:** Unprotected shared state operations result in catastrophic race conditions; while mutex locks enforce strict correctness, algorithmic work distribution with private accumulators and reductions is vastly superior for performance-critical computing.
