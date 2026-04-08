/*
 * cuBLAS Reference GEMM
 * =====================
 * cuBLAS uses column-major by default. Since our matrices are row-major,
 * we use the identity: C = A*B ↔ C^T = B^T * A^T
 * With column-major interpretation of row-major data, A becomes A^T, etc.
 * So we call: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
 *                          &alpha, B, N, A, K, &beta, C, N)
 */

#include <cublas_v2.h>

#include "../utils/cuda_utils.cuh"

void launch_cublas_gemm(cublasHandle_t handle, const float* A, const float* B, float* C, int M,
                        int N, int K) {
    float alpha = 1.0f, beta = 0.0f;
    // Row-major trick: compute C^T = B^T * A^T in column-major
    CUBLAS_CHECK(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}
