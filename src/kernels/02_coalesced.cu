/*
 * Kernel 2: Global Memory Coalescing
 * ====================================
 * Demonstrates the critical impact of memory coalescing on GPU performance.
 *
 * Key concept: Threads in a warp (32 consecutive threads by threadIdx.x)
 * should access consecutive memory addresses to allow the hardware to
 * combine (coalesce) their requests into fewer, wider memory transactions.
 *
 * We provide TWO variants:
 *   (a) UNCOALESCED: threadIdx.x → row. Warp threads write to addresses
 *       separated by stride N → each thread triggers a separate 32B transaction.
 *   (b) COALESCED: threadIdx.x → col (same as naive). Warp threads write
 *       to consecutive addresses → one 128B transaction serves 32 threads.
 *
 * On A100, the difference can be 5-10x for memory-bound kernels.
 */

#include <cuda_runtime.h>

// BAD: threadIdx.x maps to row → uncoalesced writes to C
__global__ void gemm_uncoalesced(const float * __restrict__ A,
                                  const float * __restrict__ B,
                                  float * __restrict__ C,
                                  int M, int N, int K) {
    // Deliberately swapped: x → row, y → col
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        // Threads in warp write C[row][col] where row varies → stride-N apart
        C[row * N + col] = sum;
    }
}

// GOOD: threadIdx.x maps to col → coalesced writes to C and reads from B
__global__ void gemm_coalesced(const float * __restrict__ A,
                                const float * __restrict__ B,
                                float * __restrict__ C,
                                int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // B[k*N + col]: consecutive threads read consecutive cols → coalesced
            sum += A[row * K + k] * B[k * N + col];
        }
        // C[row*N + col]: consecutive threads write consecutive cols → coalesced
        C[row * N + col] = sum;
    }
}

void launch_gemm_coalesced_stream(const float *A, const float *B, float *C,
                                  int M, int N, int K, cudaStream_t stream) {
    const int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    gemm_coalesced<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_coalesced(const float *A, const float *B, float *C,
                           int M, int N, int K) {
    launch_gemm_coalesced_stream(A, B, C, M, N, K, 0);
}

// Exported for comparison benchmarking
void launch_gemm_uncoalesced(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    const int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
    gemm_uncoalesced<<<grid, block>>>(A, B, C, M, N, K);
}
