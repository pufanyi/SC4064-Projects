/*
 * Shared Memory Tiling GEMM Kernel
 * =================================
 * Loads TILE×TILE blocks into shared memory, reusing each element TILE times.
 * Reduces global memory traffic by ~32× (TILE=32).
 */

#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int TILE_SIZE = 32;

__global__ void gemm_smem(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

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

class SmemTilingKernel : public GemmKernel {
   public:
    const char* name() const override { return "SmemTiling"; }

    void launch(const float* A, const float* B, float* C,
                int M, int N, int K, cudaStream_t stream) const override {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        gemm_smem<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

static int reg = KernelRegistry::add(std::make_unique<SmemTilingKernel>());

}  // namespace
