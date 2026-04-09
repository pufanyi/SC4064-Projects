/*
 * Tensor Parallel Linear Layer — Header
 * =======================================
 * Declarations for column/row parallel linear layers, MLP blocks,
 * and communication-compute overlap following Megatron-LM.
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "../kernels/gemm_kernel.cuh"

// ============================================================================
// Forward Passes
// ============================================================================

void column_parallel_forward(const float* d_X, const float* d_W_shard, float* d_Y_local,
                             float* d_Y_full, int M, int N, int K, int num_gpus, int gpu_id,
                             cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                             const GemmKernel& kernel);

void row_parallel_forward(const float* d_X_shard, const float* d_W_shard, float* d_Y_local,
                          float* d_Y_reduced, int M, int N, int K, int num_gpus, int gpu_id,
                          cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                          const GemmKernel& kernel);

// ============================================================================
// Backward Passes
// ============================================================================

void column_parallel_backward(const float* d_X, const float* d_W_shard, const float* d_dY_local,
                              float* d_dW_shard, float* d_dX_partial, float* d_dX,
                              float* d_W_shard_T, int M, int N, int K, int num_gpus, int gpu_id,
                              cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                              const GemmKernel& kernel);

void row_parallel_backward(const float* d_X_shard, const float* d_W_shard, const float* d_dY,
                           float* d_dW_shard, float* d_dX_shard, float* d_W_shard_T, int M, int N,
                           int K, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
                           cudaStream_t stream, const GemmKernel& kernel);

// ============================================================================
// Parallel MLP Block
// ============================================================================

void parallel_mlp_forward(const float* d_X, const float* d_W1_shard, const float* d_W2_shard,
                          float* d_hidden, float* d_Y_partial, float* d_Y, int M, int K, int H,
                          int N, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
                          cudaStream_t stream, const GemmKernel& kernel);

void parallel_mlp_backward(const float* d_X, const float* d_W1_shard, const float* d_W2_shard,
                           const float* d_hidden, const float* d_dY, float* d_dW1_shard,
                           float* d_dW2_shard, float* d_d_hidden, float* d_dX_partial, float* d_dX,
                           int M, int K, int H, int N, int num_gpus, int gpu_id,
                           cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                           const GemmKernel& kernel);

// ============================================================================
// Communication-Compute Overlap
// ============================================================================

void row_parallel_forward_overlap(const float* d_X_shard, const float* d_W_shard, float* d_Y_local,
                                  float* d_Y_reduced, int M, int N, int K, int num_gpus, int gpu_id,
                                  int num_chunks, cublasHandle_t handle, ncclComm_t comm,
                                  cudaStream_t compute_stream, cudaStream_t comm_stream,
                                  const GemmKernel& kernel);
