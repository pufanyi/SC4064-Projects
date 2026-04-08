/*
 * GEMM Kernel Dispatch
 * ====================
 * Provides a unified interface to select between custom GEMM kernels
 * and cuBLAS. This allows tensor parallel layers and benchmarks to
 * easily switch the local GEMM implementation.
 */

#pragma once

#include <cstdio>
#include <cublas_v2.h>
#include "kernels.cuh"

enum class GemmKernel {
    NAIVE = 0,
    COALESCED,
    SMEM,
    BLOCKTILE_1D,
    BLOCKTILE_2D,
    VECTORIZED,
    WARPTILE,
    CUBLAS,
    NUM_KERNELS
};

inline const char* gemm_kernel_name(GemmKernel k) {
    switch (k) {
        case GemmKernel::NAIVE:        return "1_naive";
        case GemmKernel::COALESCED:    return "2_coalesced";
        case GemmKernel::SMEM:         return "3_smem";
        case GemmKernel::BLOCKTILE_1D: return "4_1d_blocktile";
        case GemmKernel::BLOCKTILE_2D: return "5_2d_blocktile";
        case GemmKernel::VECTORIZED:   return "6_vectorized";
        case GemmKernel::WARPTILE:     return "7_warptile";
        case GemmKernel::CUBLAS:       return "cuBLAS";
        default:                       return "unknown";
    }
}

// Function pointer type for custom kernels (non-cuBLAS)
using GemmLaunchFn = void (*)(const float*, const float*, float*, int, int, int);
using GemmLaunchStreamFn =
    void (*)(const float*, const float*, float*, int, int, int, cudaStream_t);

inline GemmLaunchFn get_gemm_launch_fn(GemmKernel k) {
    switch (k) {
        case GemmKernel::NAIVE:        return launch_gemm_naive;
        case GemmKernel::COALESCED:    return launch_gemm_coalesced;
        case GemmKernel::SMEM:         return launch_gemm_smem;
        case GemmKernel::BLOCKTILE_1D: return launch_gemm_1d_blocktile;
        case GemmKernel::BLOCKTILE_2D: return launch_gemm_2d_blocktile;
        case GemmKernel::VECTORIZED:   return launch_gemm_vectorized;
        case GemmKernel::WARPTILE:     return launch_gemm_warptile;
        default:                       return nullptr;  // cuBLAS handled separately
    }
}

inline GemmLaunchStreamFn get_gemm_launch_stream_fn(GemmKernel k) {
    switch (k) {
        case GemmKernel::NAIVE:        return launch_gemm_naive_stream;
        case GemmKernel::COALESCED:    return launch_gemm_coalesced_stream;
        case GemmKernel::SMEM:         return launch_gemm_smem_stream;
        case GemmKernel::BLOCKTILE_1D: return launch_gemm_1d_blocktile_stream;
        case GemmKernel::BLOCKTILE_2D: return launch_gemm_2d_blocktile_stream;
        case GemmKernel::VECTORIZED:   return launch_gemm_vectorized_stream;
        case GemmKernel::WARPTILE:     return launch_gemm_warptile_stream;
        default:                       return nullptr;
    }
}

// Unified dispatch: runs the selected kernel
// For cuBLAS, a valid cublasHandle_t must be provided.
// For custom kernels, handle is ignored.
inline void dispatch_gemm(GemmKernel kernel,
                          const float *A, const float *B, float *C,
                          int M, int N, int K,
                          cublasHandle_t handle = nullptr) {
    if (kernel == GemmKernel::CUBLAS) {
        launch_cublas_gemm(handle, A, B, C, M, N, K);
    } else {
        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        if (fn) fn(A, B, C, M, N, K);
    }
}

inline void dispatch_gemm_on_stream(GemmKernel kernel,
                                    const float *A, const float *B, float *C,
                                    int M, int N, int K,
                                    cublasHandle_t handle,
                                    cudaStream_t stream) {
    if (kernel == GemmKernel::CUBLAS) {
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        launch_cublas_gemm(handle, A, B, C, M, N, K);
    } else {
        GemmLaunchStreamFn fn = get_gemm_launch_stream_fn(kernel);
        if (fn) fn(A, B, C, M, N, K, stream);
    }
}
