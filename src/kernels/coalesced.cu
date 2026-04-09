/*
 * Coalesced GEMM Kernel
 * =====================
 * Demonstrates memory coalescing: threadIdx.x → column for stride-1 access.
 * Also provides an uncoalesced variant for comparison.
 */

#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int BLOCK = 32;

__global__ void gemm_uncoalesced(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gemm_coalesced(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

class CoalescedKernel : public GemmKernel {
   public:
    const char* name() const override { return "Coalesced"; }

    void launch(const float* A, const float* B, float* C, int M, int N, int K,
                cudaStream_t stream) const override {
        dim3 block(BLOCK, BLOCK);
        dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
        gemm_coalesced<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

class UncoalescedKernel : public GemmKernel {
   public:
    const char* name() const override { return "Uncoalesced"; }

    void launch(const float* A, const float* B, float* C, int M, int N, int K,
                cudaStream_t stream) const override {
        dim3 block(BLOCK, BLOCK);
        dim3 grid((M + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);
        gemm_uncoalesced<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

static int reg0 = KernelRegistry::add(std::make_unique<CoalescedKernel>());
static int reg1 = KernelRegistry::add(std::make_unique<UncoalescedKernel>());

}  // namespace
