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

class CublasKernel : public GemmKernel {
    mutable cublasHandle_t handle_ = nullptr;

   public:
    const char* name() const override { return "cuBLAS"; }

    bool needs_cublas() const override { return true; }

    void set_cublas_handle(cublasHandle_t handle) override { handle_ = handle; }

    void launch(const float* A, const float* B, float* C,
                int M, int N, int K, cudaStream_t stream) const override {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle_, stream));
        CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, B, N, A, K, &beta, C, N));
    }
};

static int reg = KernelRegistry::add(std::make_unique<CublasKernel>());

}  // namespace
