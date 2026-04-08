/*
 * Kernel 3: Shared Memory Tiling
 * ===============================
 * The key bottleneck in kernels 1-2 is that every multiply requires a global
 * memory load. Global memory on A100 has ~2 TB/s bandwidth but ~400 ns latency.
 * Shared memory (on-chip SRAM, 48-164 KB per SM) has ~19 TB/s bandwidth and
 * ~20 ns latency.
 *
 * Idea: Load a TILE_SIZE × TILE_SIZE block of A and B into shared memory,
 * compute partial products, then slide the tile along K. Each element loaded
 * from global memory is reused TILE_SIZE times.
 *
 * Arithmetic intensity improvement:
 *   Before: 2 global loads per FMA = O(1) ops/byte
 *   After:  TILE_SIZE FMAs per global load = O(TILE_SIZE) ops/byte
 *   With TILE_SIZE=32: ~32x reduction in global memory traffic.
 *
 * Bank conflicts: Shared memory is divided into 32 banks (4 bytes wide).
 * Our access pattern (threadIdx.x → column) avoids bank conflicts for B,
 * and broadcasts for A (same row accessed by all threads in a row).
 */

#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void gemm_smem(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Slide tile along K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative load: each thread loads one element of each tile
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        // Bounds checking for non-multiple-of-TILE_SIZE dimensions
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

// Compute partial dot product from this tile
#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_gemm_smem_stream(const float* A, const float* B, float* C, int M, int N, int K,
                             cudaStream_t stream) {
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    gemm_smem<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_smem(const float* A, const float* B, float* C, int M, int N, int K) {
    launch_gemm_smem_stream(A, B, C, M, N, K, 0);
}
