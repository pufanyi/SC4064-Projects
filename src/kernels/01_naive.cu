/*
 * Kernel 1: Naive GEMM
 * ====================
 * Each thread computes exactly one element of C.
 *
 * Performance characteristics:
 *   - Every element of C requires K global memory loads from A and K from B.
 *   - No data reuse whatsoever — entirely memory-bound.
 *   - A is accessed row-wise (stride-1 within a thread, but across threads in
 *     a warp the access to A[row][k] is the SAME address → broadcast, OK).
 *   - B is accessed column-wise: threads in a warp access B[k][col],
 *     where col varies by threadIdx.x → stride-1, coalesced.
 *   - However the thread mapping (row = blockIdx.y, col = blockIdx.x) means
 *     that we get coalesced reads on B but NOT on the output C write pattern
 *     depending on block dimensions.
 *
 * Expected: ~1-2% of cuBLAS performance for large matrices.
 */

#include <cuda_runtime.h>

__global__ void gemm_naive(const float * __restrict__ A,
                           const float * __restrict__ B,
                           float * __restrict__ C,
                           int M, int N, int K) {
    // Map thread to output element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_gemm_naive_stream(const float *A, const float *B, float *C,
                              int M, int N, int K, cudaStream_t stream) {
    const int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    gemm_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_naive(const float *A, const float *B, float *C,
                       int M, int N, int K) {
    launch_gemm_naive_stream(A, B, C, M, N, K, 0);
}
