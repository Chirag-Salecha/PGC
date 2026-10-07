# Topic 5: Distributed Vector Processing
## Comprehensive Viva Voce Questions & Answers (25 Key Questions)

This guide contains simple, clear, and direct answers to the most common viva questions asked in lab evaluations for Parallel Computing and MPI.

---

### Q1. What is parallel computing?
**Answer:** Parallel computing is a type of computation where multiple processing units (CPU cores, processors, or computers) perform multiple calculations simultaneously. Large problems are broken down into smaller, independent sub-problems that are solved at the same time to save time.

---

### Q2. What is MPI?
**Answer:** MPI stands for **Message Passing Interface**. It is a standardized and portable message-passing system designed for parallel computing on distributed-memory computer systems. It allows independent processes to communicate by sending and receiving messages.

---

### Q3. Why do we use MPI?
**Answer:**
1. **Scalability:** MPI programs can run on a single multi-core laptop or scale to thousands of nodes across a supercomputer cluster.
2. **Distributed Memory:** Unlike OpenMP, which is limited to the RAM on a single machine, MPI allows processes running on different machines across a network to collaborate.
3. **Portability:** MPI is supported on virtually all parallel architectures.

---

### Q4. What is a process?
**Answer:** A process is an independent program in execution. In MPI, each process has its own private memory address space. One process cannot directly access or modify the variables of another process without sending a message.

---

### Q5. What is an MPI Rank?
**Answer:** A rank is a unique integer identifier assigned to each process within an MPI communicator, starting from `0` up to `P - 1` (where $P$ is the total number of processes).
* Rank 0 is typically designated as the **Root** or **Master** process.

---

### Q6. What is an MPI Communicator?
**Answer:** An MPI communicator is a communication domain that groups a collection of processes together. The default predefined communicator containing all processes is **`MPI_COMM_WORLD`**. All collective communication takes place within a specified communicator.

---

### Q7. What is `MPI_Scatter`?
**Answer:** `MPI_Scatter` is a collective communication function that takes a large array from one root process (usually rank 0), splits it into equal segments, and sends one segment to each process in the communicator, including itself.

---

### Q8. What is `MPI_Gather`?
**Answer:** `MPI_Gather` is the reverse of `MPI_Scatter`. It collects individual data chunks from all processes in the communicator and concatenates them in order into a single large array on the root process.

---

### Q9. What is `MPI_Reduce`?
**Answer:** `MPI_Reduce` collects data from all processes, applies an associative mathematical reduction operation (such as `MPI_SUM`, `MPI_MAX`, or `MPI_MIN`), and returns the single combined scalar result to the root process.

---

### Q10. What is `MPI_Bcast` (Broadcast)?
**Answer:** `MPI_Bcast` sends the exact same piece of data from the root process to every other process in the communicator.
* *Difference:* `MPI_Bcast` sends the *same whole buffer* to all processes; `MPI_Scatter` divides the buffer and sends *different portions* to each process.

---

### Q11. What is `MPI_Barrier`?
**Answer:** `MPI_Barrier` is a synchronization collective. When called, no process in the communicator can proceed past the barrier until all processes have reached the barrier. It is primarily used to align processes before and after benchmarking.

---

### Q12. What is `MPI_Wtime`?
**Answer:** `MPI_Wtime()` returns the current elapsed wall-clock time in seconds as a high-resolution double-precision floating-point number. It is used to measure execution time.

---

### Q13. What is synchronization in parallel computing?
**Answer:** Synchronization is the coordination of concurrent processes to ensure they reach a specific execution point or state before continuing. In MPI, barrier synchronization prevents fast processes from racing ahead of slow processes.

---

### Q14. Why can parallel execution be faster than sequential execution?
**Answer:** Parallel execution divides the total workload among multiple CPU cores. Instead of 1 core doing $10^7$ calculations sequentially, 4 cores can each perform $2.5 \times 10^6$ calculations simultaneously, reducing the total wall-clock time.

---

### Q15. What is Speedup?
**Answer:** Speedup ($S_P$) measures how much faster a parallel algorithm runs compared to the sequential algorithm:
$$\text{Speedup } (S_P) = \frac{\text{Sequential Execution Time } (T_1)}{\text{Parallel Execution Time } (T_P)}$$
If sequential time is 4.0s and parallel time is 1.0s, Speedup = 4.0x.

---

### Q16. What is Parallel Efficiency?
**Answer:** Efficiency ($E_P$) measures how effectively the allocated CPU cores are being utilized:
$$\text{Efficiency } (E_P) = \frac{\text{Speedup } (S_P)}{P} = \frac{T_1}{P \times T_P}$$
Ideal efficiency is $1.0$ (or 100%).

---

### Q17. Why does Speedup not increase linearly ($S_P < P$)?
**Answer:**
1. **Communication Overhead:** Time spent sending and receiving data over the network or IPC (`MPI_Scatter` and `MPI_Gather`).
2. **Amdahl's Law:** Every program has non-parallelizable serial portions (e.g., memory allocation, initialization, reading files).
3. **Load Imbalance & Synchronization Delays:** Processes waiting for the slowest process to reach a barrier.

---

### Q18. What is communication overhead?
**Answer:** Communication overhead is the time and computational cost spent on packaging, transmitting, synchronizing, and unpacking data between processes, rather than doing useful arithmetic computation.

---

### Q19. What happens when the number of processes increases excessively?
**Answer:** If we increase processes too much for a fixed problem size:
* The computation chunk per process becomes very small.
* The communication time to scatter and gather begins to exceed the computation time.
* Speedup plateaus or even decreases, causing parallel efficiency to drop sharply.

---

### Q20. What is the fundamental difference between sequential and parallel execution?
**Answer:**
* **Sequential Execution:** Instructions execute one after another in a single sequence on a single processor core with a single memory space.
* **Parallel Execution:** Multiple processors execute distinct subsets of instructions simultaneously, sharing memory (OpenMP) or exchanging messages across private memories (MPI).

---

### Q21. What is Domain Decomposition (Data Parallelism)?
**Answer:** Domain decomposition is the technique of dividing the data space (e.g., an array, grid, or vector) into sub-domains and assigning each sub-domain to a distinct process to compute concurrently.

---

### Q22. What happens if a process fails to call `MPI_Finalize()`?
**Answer:** If `MPI_Finalize()` is omitted, the MPI runtime environment will not cleanly disconnect sockets, release shared memory resources, or terminate background daemons, often causing the application to hang or terminate with resource leak errors.

---

### Q23. What is the difference between Point-to-Point and Collective communication?
**Answer:**
* **Point-to-Point (`MPI_Send`, `MPI_Recv`):** Communication between exactly two specific processes (sender and receiver).
* **Collective (`MPI_Bcast`, `MPI_Scatter`, `MPI_Gather`, `MPI_Reduce`):** Communication involving all processes within a communicator simultaneously.

---

### Q24. How did you verify the correctness of your MPI program?
**Answer:** We computed the exact same mathematical formula on the identical dataset in a sequential reference program. We compared:
1. The global scalar sum reduction ($|S_{\text{seq}} - S_{\text{mpi}}| < 10^{-5}$).
2. The element-wise output values across the array.
When both matched, our test reported `PASS`.

---

### Q25. What is the difference between MPI and OpenMP?
**Answer:**
* **OpenMP:** Shared-memory model, thread-based, uses compiler directives (`#pragma omp parallel`), limited to single computer/node.
* **MPI:** Distributed-memory model, process-based, uses explicit library function calls, scales across networks and supercomputers.
