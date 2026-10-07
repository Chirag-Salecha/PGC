/**
 * ============================================================================
 * Distributed Vector Processing - Sequential Baseline
 * File: src/sequential.c
 * Topic: 5 (Parallel Computing Mini-Project)
 *
 * Description:
 *   Performs element-wise vector transformation and sum reduction on a single
 *   CPU core. This serves as the reference benchmark to measure parallel
 *   speedup and efficiency and to verify the correctness of the MPI implementation.
 *
 * Mathematical Kernel:
 *   Y[i] = sqrt(X[i] * X[i] + 2.5 * X[i] + 1.0)
 *   Sum  = \sum Y[i]
 * ============================================================================
 */

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* Helper function: Get current time in seconds */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    /* Default vector size: 10 million elements */
    long long N = 10000000LL;
    char *dump_file = NULL;

    /* Parse command line arguments: ./sequential [N] [optional_output_dump] */
    if (argc >= 2) {
        N = atoll(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Error: Vector size N must be positive.\n");
            return 1;
        }
    }
    if (argc >= 3) {
        dump_file = argv[2];
    }

    printf("=====================================================\n");
    printf("   TOPIC 5: SEQUENTIAL VECTOR PROCESSING (BASELINE)  \n");
    printf("=====================================================\n");
    printf("Vector Size (N)     : %lld elements\n", N);
    printf("Memory Required     : %.2f MB (Input + Output)\n", 
           (double)(2 * N * sizeof(double)) / (1024.0 * 1024.0));

    /* 1. Allocate memory for input vector X and output vector Y */
    double *X = (double *)malloc(N * sizeof(double));
    double *Y = (double *)malloc(N * sizeof(double));

    if (X == NULL || Y == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for size %lld.\n", N);
        if (X) free(X);
        if (Y) free(Y);
        return 1;
    }

    /* 2. Initialize input vector */
    /* Using deterministic values so MPI and Sequential run identical data */
    for (long long i = 0; i < N; i++) {
        X[i] = (double)(i + 1) * 0.001;
    }

    /* 3. Execute and time the sequential computation */
    printf("\nComputing element-wise transformation sequentially...\n");
    double start_time = get_time_sec();

    double sum = 0.0;
    for (long long i = 0; i < N; i++) {
        /* Compute Y[i] = sqrt(X[i]^2 + 2.5 * X[i] + 1.0) */
        double val = X[i];
        double res = sqrt(val * val + 2.5 * val + 1.0);
        Y[i] = res;
        sum += res;
    }

    double end_time = get_time_sec();
    double elapsed_time = end_time - start_time;

    /* 4. Display Results */
    printf("Computation finished successfully!\n");
    printf("-----------------------------------------------------\n");
    printf("Sequential Execution Time : %.6f seconds\n", elapsed_time);
    printf("Verification Checksum (Sum): %.6f\n", sum);
    printf("First 5 Results           : [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           Y[0], Y[1], Y[2], Y[3], Y[4]);
    printf("Last 5 Results            : [%.4f, %.4f, %.4f, %.4f, %.4f]\n",
           Y[N-5], Y[N-4], Y[N-3], Y[N-2], Y[N-1]);
    printf("-----------------------------------------------------\n");

    /* Optionally dump first/last elements or binary checksum for validation */
    if (dump_file != NULL) {
        FILE *fp = fopen(dump_file, "w");
        if (fp != NULL) {
            fprintf(fp, "%lld\n", N);
            fprintf(fp, "%.10f\n", sum);
            fprintf(fp, "%.10f\n", elapsed_time);
            /* Dump first 10 and last 10 elements for verification */
            for (int i = 0; i < 10 && i < N; i++) {
                fprintf(fp, "%.10f\n", Y[i]);
            }
            for (long long i = (N > 10 ? N - 10 : 0); i < N; i++) {
                fprintf(fp, "%.10f\n", Y[i]);
            }
            fclose(fp);
            printf("Saved verification data to: %s\n", dump_file);
        }
    }

    /* 5. Clean up allocated memory */
    free(X);
    free(Y);

    return 0;
}
