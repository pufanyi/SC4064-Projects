/*
 * Naive GEMM Kernel
 * =================
 * Each thread computes exactly one element of C.
 * No data reuse — entirely memory-bound.
 * Expected: ~1-2% of cuBLAS for large matrices.
 */

#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

__global__ void gemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
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

class NaiveKernel : public GemmKernel {
   public:
    const char* name() const override { return "Naive"; }

    void launch(const float* A, const float* B, float* C, int M, int N, int K,
                cudaStream_t stream) const override {
        constexpr int BLOCK = 32;
        dim3 block(BLOCK, BLOCK);
        dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
        gemm_naive<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

static int reg = KernelRegistry::add(std::make_unique<NaiveKernel>());

}  // namespace
