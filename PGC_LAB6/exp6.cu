#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE 32
#define BLK 16


#define CHECK(call) do { \
    cudaError_t checkErr = (call); \
    if (checkErr != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(checkErr), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

// ---------- Kernel 1: UNCOALESCED transpose ----------
__global__ void transposeNaive(const float *in, float *out, int N)
{
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < N)
        out[x * N + y] = in[y * N + x];
}


// ---------- Kernel 2: COALESCED transpose ----------
__global__ void transposeCoalesced(const float *in, float *out, int N)
{
    // Shared memory with padding to avoid bank conflicts.
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // Coalesced global memory read.
    if (x < N && y < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];

    // Synchronize threads before accessing the tile.
    __syncthreads();

    // Swap block coordinates.
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // Coalesced global memory write.
    if (x < N && y < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}


// ---------- Kernel 3: NAIVE matrix multiplication ----------
// C = A x B

__global__ void matMulNaive(
    const float *A,
    const float *B,
    float *C,
    int N)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// ---------- Host helpers: timing and verification ----------

typedef void (*TKernel)(const float *, float *, int);

// Measure average transpose execution time
float timeTranspose(
    TKernel k,
    dim3 g,
    dim3 b,
    const float *in,
    float *out,
    int N,
    int reps)
{
    cudaEvent_t s, e;

    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    CHECK(cudaMemset(
        out, 0, (size_t)N * N * sizeof(float)));

    // Warm-up execution
    k<<<g, b>>>(in, out, N);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Start GPU timing
    CHECK(cudaEventRecord(s));

    for (int r = 0; r < reps; r++) {
        k<<<g, b>>>(in, out, N);
    }

    // Stop GPU timing
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));

    float ms;

    CHECK(cudaEventElapsedTime(&ms, s, e));

    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));

    return ms / reps;
}


// Verify matrix transpose output
bool checkTranspose(
    const float *in,
    const float *out,
    int N)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            if (out[j * N + i] != in[i * N + j]) {
                return false;
            }
        }
    }

    return true;
}


// Verify matrix multiplication output
bool checkMatMul(
    const float *A,
    const float *B,
    const float *C,
    int N)
{
    // Check 20 randomly selected elements
    for (int t = 0; t < 20; t++) {

        int i = rand() % N;
        int j = rand() % N;

        double ref = 0.0;

        for (int k = 0; k < N; k++) {
            ref += (double)A[i * N + k]
                 * B[k * N + j];
        }

        if (fabs(ref - C[i * N + j])
            > 1e-3 * fabs(ref) + 1e-3) {

            return false;
        }
    }

    return true;
}



// ---------- MAIN FUNCTION ----------

int main(int argc, char **argv)
{
    int NT = 4096;  // Transpose matrix size
    int NM = 1024;  // Matrix multiplication size

    if (argc > 1) NT = atoi(argv[1]);
    if (argc > 2) NM = atoi(argv[2]);

    // Display GPU information
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s (compute capability %d.%d)\n",
           prop.name, prop.major, prop.minor);

    // =========================================
    // PART A: MATRIX TRANSPOSE
    // =========================================

    size_t bytesT = (size_t)NT * NT * sizeof(float);

    float *h_in = (float *)malloc(bytesT);
    float *h_out = (float *)malloc(bytesT);

    // Initialize input matrix
    for (size_t i = 0; i < (size_t)NT * NT; i++)
        h_in[i] = (float)(i % 1000);

    // Allocate GPU memory
    float *d_in, *d_out;

    CHECK(cudaMalloc(&d_in, bytesT));
    CHECK(cudaMalloc(&d_out, bytesT));

    // Copy input to GPU
    CHECK(cudaMemcpy(d_in, h_in, bytesT,
                     cudaMemcpyHostToDevice));

    // Define CUDA grid and block dimensions
    dim3 blockT(TILE, TILE);

    dim3 gridT(
        (NT + TILE - 1) / TILE,
        (NT + TILE - 1) / TILE
    );

    // Run uncoalesced transpose
    float tN = timeTranspose(
        transposeNaive,
        gridT, blockT,
        d_in, d_out, NT, 20
    );

    CHECK(cudaMemcpy(h_out, d_out, bytesT,
                     cudaMemcpyDeviceToHost));

    bool okN = checkTranspose(h_in, h_out, NT);

    // Run coalesced transpose
    float tC = timeTranspose(
        transposeCoalesced,
        gridT, blockT,
        d_in, d_out, NT, 20
    );

    CHECK(cudaMemcpy(h_out, d_out, bytesT,
                     cudaMemcpyDeviceToHost));

    bool okC = checkTranspose(h_in, h_out, NT);

    // Calculate effective memory bandwidth
    double gbN = 2.0 * bytesT / (tN * 1e6);
    double gbC = 2.0 * bytesT / (tC * 1e6);

    // Display transpose performance
    printf("\n=== PART A: Matrix transpose (N = %d) ===\n", NT);

    printf("Uncoalesced : %8.3f ms | %7.1f GB/s | %s\n",
           tN, gbN, okN ? "PASS" : "FAIL");

    printf("Coalesced   : %8.3f ms | %7.1f GB/s | %s\n",
           tC, gbC, okC ? "PASS" : "FAIL");

    printf("Speedup (uncoalesced / coalesced) = %.2fx\n",
           tN / tC);


    // =========================================
    // PART B: NAIVE MATRIX MULTIPLICATION
    // =========================================

    size_t bytesM = (size_t)NM * NM * sizeof(float);

    float *hA = (float *)malloc(bytesM);
    float *hB = (float *)malloc(bytesM);
    float *hC = (float *)malloc(bytesM);

    // Initialize matrices A and B
    for (size_t i = 0; i < (size_t)NM * NM; i++) {
        hA[i] = rand() / (float)RAND_MAX;
        hB[i] = rand() / (float)RAND_MAX;
    }

    // Allocate GPU memory
    float *dA, *dB, *dC;

    CHECK(cudaMalloc(&dA, bytesM));
    CHECK(cudaMalloc(&dB, bytesM));
    CHECK(cudaMalloc(&dC, bytesM));

    // Copy matrices to GPU
    CHECK(cudaMemcpy(dA, hA, bytesM,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dB, hB, bytesM,
                     cudaMemcpyHostToDevice));

    // Set grid and block dimensions
    dim3 block(BLK, BLK);

    dim3 grid(
        (NM + BLK - 1) / BLK,
        (NM + BLK - 1) / BLK
    );

    // CUDA timing events
    cudaEvent_t s, e;

    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));

    // GPU warm-up
    matMulNaive<<<grid, block>>>(dA, dB, dC, NM);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // Measure matrix multiplication time
    int reps = 5;

    CHECK(cudaEventRecord(s));

    for (int r = 0; r < reps; r++)
        matMulNaive<<<grid, block>>>(dA, dB, dC, NM);

    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));

    float ms;

    CHECK(cudaEventElapsedTime(&ms, s, e));

    float tM = ms / reps;

    // Copy results back to CPU
    CHECK(cudaMemcpy(hC, dC, bytesM,
                     cudaMemcpyDeviceToHost));

    // Verify multiplication results
    bool okM = checkMatMul(hA, hB, hC, NM);

    // Calculate GFLOPS
    double gflops =
        2.0 * NM * NM * (double)NM / (tM * 1e6);

    printf("\n=== PART B: Naive matrix multiplication (N = %d) ===\n",
           NM);

    printf("Naive matmul: %8.3f ms | %7.1f GFLOPS | %s\n",
           tM, gflops, okM ? "PASS" : "FAIL");


    // =========================================
    // FINAL SUMMARY
    // =========================================

    printf("\n=== SUMMARY ===\n");

    printf("Transpose uncoalesced : %.3f ms\n", tN);
    printf("Transpose coalesced   : %.3f ms\n", tC);
    printf("Naive matmul          : %.3f ms\n", tM);

    // Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Free CPU memory
    free(h_in);
    free(h_out);
    free(hA);
    free(hB);
    free(hC);

    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));

    return (okN && okC && okM) ? 0 : 1;
}


