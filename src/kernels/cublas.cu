/*
 * cuBLAS GEMM Kernel
 * ==================
 * Wraps cublasSgemm with the row-major → column-major trick:
 *   C = A×B  ↔  C^T = B^T × A^T
 */

#include <cublas_v2.h>

#include "../utils/cuda_utils.cuh"
#include "kernel_registry.cuh"

namespace {

// Free thread_local to avoid potential NVCC issues with class-member thread_local.
static thread_local cublasHandle_t g_cublas_handle = nullptr;

class CublasKernel : public GemmKernel {
   public:
    const char* name() const override { return "cuBLAS"; }

    bool needs_cublas() const override { return true; }

    void set_cublas_handle(cublasHandle_t handle) override { g_cublas_handle = handle; }

    void launch(const float* A, const float* B, float* C,
                int M, int N, int K, cudaStream_t stream) const override {
        if (!g_cublas_handle) return;
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(g_cublas_handle, stream));
        CUBLAS_CHECK(cublasSgemm(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, N, A, K, &beta, C, N));
    }
};


static int reg = KernelRegistry::add(std::make_unique<CublasKernel>());

}  // namespace
