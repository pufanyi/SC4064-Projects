#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

// All kernels compute C = A * B in row-major layout
// A: MxK, B: KxN, C: MxN

// Kernel 1: Naive — one thread per output element, uncoalesced A access
void launch_gemm_naive(const float *A, const float *B, float *C,
                       int M, int N, int K);
void launch_gemm_naive_stream(const float *A, const float *B, float *C,
                              int M, int N, int K, cudaStream_t stream);

// Kernel 2: Global memory coalescing — transpose loop order for coalesced B reads
void launch_gemm_coalesced(const float *A, const float *B, float *C,
                           int M, int N, int K);
void launch_gemm_coalesced_stream(const float *A, const float *B, float *C,
                                  int M, int N, int K, cudaStream_t stream);

// Kernel 3: Shared memory tiling — TILE_SIZE x TILE_SIZE tiles in smem
void launch_gemm_smem(const float *A, const float *B, float *C,
                      int M, int N, int K);
void launch_gemm_smem_stream(const float *A, const float *B, float *C,
                             int M, int N, int K, cudaStream_t stream);

// Kernel 4: 1D block tiling — each thread computes TM elements along M
void launch_gemm_1d_blocktile(const float *A, const float *B, float *C,
                              int M, int N, int K);
void launch_gemm_1d_blocktile_stream(const float *A, const float *B, float *C,
                                     int M, int N, int K, cudaStream_t stream);

// Kernel 5: 2D block tiling — each thread computes TM x TN sub-tile in registers
void launch_gemm_2d_blocktile(const float *A, const float *B, float *C,
                              int M, int N, int K);
void launch_gemm_2d_blocktile_stream(const float *A, const float *B, float *C,
                                     int M, int N, int K, cudaStream_t stream);

// Kernel 6: Vectorized loads — uses float4 for 128-bit memory transactions
void launch_gemm_vectorized(const float *A, const float *B, float *C,
                            int M, int N, int K);
void launch_gemm_vectorized_stream(const float *A, const float *B, float *C,
                                   int M, int N, int K, cudaStream_t stream);

// Kernel 7: Warp tiling — hierarchical block→warp→thread tiling
void launch_gemm_warptile(const float *A, const float *B, float *C,
                          int M, int N, int K);
void launch_gemm_warptile_stream(const float *A, const float *B, float *C,
                                 int M, int N, int K, cudaStream_t stream);

// cuBLAS reference
void launch_cublas_gemm(cublasHandle_t handle,
                        const float *A, const float *B, float *C,
                        int M, int N, int K);
