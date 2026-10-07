/**
 * ============================================================================
 * Distributed Vector Processing - MPI Parallel Implementation
 * File: src/mpi_program.c
 * Topic: 5 (Parallel Computing Mini-Project)
 *
 * Description:
 *   Distributes a large vector across multiple MPI processes, performs
 *   independent element-wise computations in parallel, and gathers the results
 *   back to the root process.
 *
 * Key MPI Collectives Used:
 *   - MPI_Init / MPI_Finalize     : Lifecycle management
 *   - MPI_Comm_size / MPI_Comm_rank: Environment queries
 *   - MPI_Scatter                 : Distributes chunks of vector X to all processes
 *   - MPI_Gather                  : Collects computed chunks Y back to process 0
 *   - MPI_Reduce                  : Computes global reduction sum of elements
 *   - MPI_Barrier                 : Synchronizes all processes for accurate timing
 *   - MPI_Wtime                   : High-resolution wall-clock timer
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, num_procs;

    /* 1. Initialize MPI Execution Environment */
    MPI_Init(&argc, &argv);

    /* 2. Determine total number of processes and current process rank */
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Parse vector size N from command line arguments (default: 10,000,000) */
    long long N = 10000000LL;
    char *dump_file = NULL;
    int csv_mode = 0;

    if (argc >= 2) {
        N = atoll(argv[1]);
    }
    if (argc >= 3) {
        if (strcmp(argv[2], "--csv") == 0) {
            csv_mode = 1;
        } else {
            dump_file = argv[2];
        }
    }
    if (argc >= 4 && strcmp(argv[3], "--csv") == 0) {
        csv_mode = 1;
    }

    /* Ensure N is evenly divisible by number of processes */
    if (N % num_procs != 0) {
        long long old_N = N;
        N = (N / num_procs) * num_procs;
        if (rank == 0 && !csv_mode) {
            printf("[Notice] Adjusted vector size N from %lld to %lld to divide evenly among %d processes.\n",
                   old_N, N, num_procs);
        }
    }

    /* Local chunk size assigned to each MPI process */
    long long local_n = N / num_procs;

    /* Pointers for global buffers (only allocated on Process 0) */
    double *global_X = NULL;
    double *global_Y = NULL;

    if (rank == 0) {
        if (!csv_mode) {
            printf("=====================================================\n");
            printf("      TOPIC 5: MPI DISTRIBUTED VECTOR PROCESSING     \n");
            printf("=====================================================\n");
            printf("MPI Processes (P)   : %d\n", num_procs);
            printf("Total Elements (N)  : %lld\n", N);
            printf("Elements per Process: %lld\n", local_n);
            printf("Memory on Root Node : %.2f MB\n", 
                   (double)(2 * N * sizeof(double)) / (1024.0 * 1024.0));
            printf("-----------------------------------------------------\n");
        }

        /* Root allocates full input and output vectors */
        global_X = (double *)malloc(N * sizeof(double));
        global_Y = (double *)malloc(N * sizeof(double));

        if (global_X == NULL || global_Y == NULL) {
            fprintf(stderr, "Rank 0 Error: Memory allocation failed for global vectors.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Root initializes input data identically to the sequential baseline */
        for (long long i = 0; i < N; i++) {
            global_X[i] = (double)(i + 1) * 0.001;
        }
    }

    /* 3. Each process allocates local buffers for its chunk */
    double *local_X = (double *)malloc(local_n * sizeof(double));
    double *local_Y = (double *)malloc(local_n * sizeof(double));

    if (local_X == NULL || local_Y == NULL) {
        fprintf(stderr, "Rank %d Error: Memory allocation failed for local chunk.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Synchronize before timing to ensure all processes start together */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();

    /* -------------------------------------------------------------
     * PHASE 1: SCATTER INPUT VECTOR CHUNKS TO ALL PROCESSES
     * ------------------------------------------------------------- */
    double t_scatter_start = MPI_Wtime();
    MPI_Scatter(global_X, local_n, MPI_DOUBLE,
                local_X,  local_n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    double t_scatter_end = MPI_Wtime();

    /* -------------------------------------------------------------
     * PHASE 2: INDEPENDENT LOCAL COMPUTATION (PARALLEL EXECUTION)
     * ------------------------------------------------------------- */
    double t_comp_start = MPI_Wtime();
    double local_sum = 0.0;

    for (long long j = 0; j < local_n; j++) {
        double val = local_X[j];
        /* Element-wise formula: Y = sqrt(X^2 + 2.5*X + 1.0) */
        double res = sqrt(val * val + 2.5 * val + 1.0);
        local_Y[j] = res;
        local_sum += res;
    }
    double t_comp_end = MPI_Wtime();
    double local_comp_time = t_comp_end - t_comp_start;

    /* -------------------------------------------------------------
     * PHASE 3: GATHER COMPUTED CHUNKS BACK TO PROCESS 0
     * ------------------------------------------------------------- */
    double t_gather_start = MPI_Wtime();
    MPI_Gather(local_Y,  local_n, MPI_DOUBLE,
               global_Y, local_n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    double t_gather_end = MPI_Wtime();

    /* -------------------------------------------------------------
     * PHASE 4: GLOBAL REDUCTION (SUM ACROSS ALL PROCESSES)
     * ------------------------------------------------------------- */
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Synchronize before stopping the timer */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_end = MPI_Wtime();

    /* Calculate timings on root process */
    double total_wall_time = t_total_end - t_total_start;
    double comm_time = (t_scatter_end - t_scatter_start) + (t_gather_end - t_gather_start);

    /* Find maximum computation time across processes (load balance check) */
    double max_comp_time = 0.0;
    MPI_Reduce(&local_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* -------------------------------------------------------------
     * PHASE 5: REPORT RESULTS & TIMINGS
     * ------------------------------------------------------------- */
    if (rank == 0) {
        if (csv_mode) {
            /* Output format for automated scripting:
             * ProcessCount,VectorSize,TotalTime,CompTime,CommTime,Checksum
             */
            printf("%d,%lld,%.6f,%.6f,%.6f,%.6f\n",
                   num_procs, N, total_wall_time, max_comp_time, comm_time, global_sum);
        } else {
            printf("MPI Distributed Computation Complete!\n");
            printf("-----------------------------------------------------\n");
            printf("Total Parallel Time (Wall): %.6f seconds\n", total_wall_time);
            printf("Max Local Compute Time    : %.6f seconds\n", max_comp_time);
            printf("Scatter + Gather Comm Time: %.6f seconds\n", comm_time);
            printf("Verification Checksum(Sum): %.6f\n", global_sum);
            printf("First 5 Results           : [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                   global_Y[0], global_Y[1], global_Y[2], global_Y[3], global_Y[4]);
            printf("Last 5 Results            : [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
                   global_Y[N-5], global_Y[N-4], global_Y[N-3], global_Y[N-2], global_Y[N-1]);
            printf("-----------------------------------------------------\n");
        }

        /* Optionally save verification dump */
        if (dump_file != NULL) {
            FILE *fp = fopen(dump_file, "w");
            if (fp != NULL) {
                fprintf(fp, "%lld\n", N);
                fprintf(fp, "%.10f\n", global_sum);
                fprintf(fp, "%.10f\n", total_wall_time);
                for (int i = 0; i < 10 && i < N; i++) {
                    fprintf(fp, "%.10f\n", global_Y[i]);
                }
                for (long long i = (N > 10 ? N - 10 : 0); i < N; i++) {
                    fprintf(fp, "%.10f\n", global_Y[i]);
                }
                fclose(fp);
                printf("Saved MPI verification data to: %s\n", dump_file);
            }
        }

        free(global_X);
        free(global_Y);
    }

    /* 6. Clean up local memory and finalize MPI */
    free(local_X);
    free(local_Y);

    MPI_Finalize();
    return 0;
}
