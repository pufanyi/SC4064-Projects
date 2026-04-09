/*
 * Tensor Parallel Linear Layer with NCCL
 * ========================================
 * Implements Column Parallel and Row Parallel linear layers following
 * Megatron-LM (Shoeybi et al., 2019) for distributed GEMM across GPUs.
 *
 * Column Parallelism:
 *   Weight W is split column-wise: W = [W_0 | W_1 | ... | W_{p-1}]
 *   Each GPU i computes: Y_i = X @ W_i    (partial output)
 *   To get full output: AllGather(Y_0, Y_1, ..., Y_{p-1})
 *
 * Row Parallelism:
 *   Weight W is split row-wise: W = [W_0; W_1; ...; W_{p-1}]
 *   Input X is split column-wise: X = [X_0 | X_1 | ... | X_{p-1}]
 *   Each GPU i computes: Y_i = X_i @ W_i  (partial sum)
 *   To get full output: AllReduce(Y_0 + Y_1 + ... + Y_{p-1})
 *
 * Together, column -> row parallelism forms a complete MLP block
 * with only ONE AllReduce per forward pass and ONE per backward pass.
 */

#include "tensor_parallel.cuh"

#include "../utils/cuda_raii.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/nccl_utils.cuh"

// ============================================================================
// Helpers
// ============================================================================

// Transpose matrix on GPU via cuBLAS (out-of-place): B[N×M] = A[M×N]^T
static void gpu_transpose(cublasHandle_t handle, const float* A, float* B, int M, int N,
                          cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N,
                             &alpha, A, N, &beta, B, M, B, M));
}

// Local GEMM via kernel abstraction
static void local_gemm(const float* A, const float* B, float* C, int M, int N, int K,
                       const GemmKernel& kernel, cudaStream_t stream) {
    kernel.launch(A, B, C, M, N, K, stream);
}

// Gradient GEMM helper: handles the cuBLAS transpose path vs custom kernel path.
// Computes C[M×N] = A^T[M×K_orig] @ B[K_orig×N]  (i.e., A is K_orig×M, transposed)
static void grad_gemm_at_b(const float* A, const float* B, float* C,
                           int M, int N, int K_orig,
                           const GemmKernel& kernel, cublasHandle_t handle,
                           cudaStream_t stream) {
    if (kernel.needs_cublas()) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        // C = A^T @ B  in row-major via col-major trick
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 N, M, K_orig, &alpha,
                                 B, N, A, M, &beta, C, N));
    } else {
        CudaMemory<float> A_T(M * K_orig);
        gpu_transpose(handle, A, A_T.get(), K_orig, M, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        kernel.launch(A_T.get(), B, C, M, N, K_orig);
    }
}

// Gradient GEMM helper: C[M×N] = A[M×K] @ B^T[K×N_orig]  (B is N_orig×K, transposed)
static void grad_gemm_a_bt(const float* A, const float* B, float* C,
                           int M, int N, int K,
                           const GemmKernel& kernel, cublasHandle_t handle,
                           float* B_T_workspace, cudaStream_t stream) {
    if (kernel.needs_cublas()) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, M, K, &alpha,
                                 B, K, A, K, &beta, C, N));
    } else {
        gpu_transpose(handle, B, B_T_workspace, N, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        kernel.launch(A, B_T_workspace, C, M, N, K);
    }
}

// ============================================================================
// FORWARD PASSES
// ============================================================================

void column_parallel_forward(const float* d_X, const float* d_W_shard, float* d_Y_local,
                             float* d_Y_full, int M, int N, int K, int num_gpus, int gpu_id,
                             cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                             const GemmKernel& kernel) {
    int N_local = N / num_gpus;
    local_gemm(d_X, d_W_shard, d_Y_local, M, N_local, K, kernel, stream);

    if (d_Y_full) {
        if (num_gpus == 1) {
            if (d_Y_full != d_Y_local) {
                CUDA_CHECK(cudaMemcpyAsync(d_Y_full, d_Y_local, M * N_local * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
            }
        } else {
            NCCL_CHECK(ncclAllGather(d_Y_local, d_Y_full, M * N_local, ncclFloat, comm, stream));
        }
    }
}

void row_parallel_forward(const float* d_X_shard, const float* d_W_shard, float* d_Y_local,
                          float* d_Y_reduced, int M, int N, int K, int num_gpus, int gpu_id,
                          cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                          const GemmKernel& kernel) {
    int K_local = K / num_gpus;
    local_gemm(d_X_shard, d_W_shard, d_Y_local, M, N, K_local, kernel, stream);

    if (num_gpus == 1) {
        if (d_Y_reduced != d_Y_local) {
            CUDA_CHECK(cudaMemcpyAsync(d_Y_reduced, d_Y_local, M * N * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        NCCL_CHECK(ncclAllReduce(d_Y_local, d_Y_reduced, M * N, ncclFloat, ncclSum, comm, stream));
    }
}

// ============================================================================
// BACKWARD PASSES
// ============================================================================

void column_parallel_backward(const float* d_X, const float* d_W_shard, const float* d_dY_local,
                              float* d_dW_shard, float* d_dX_partial, float* d_dX,
                              float* d_W_shard_T, int M, int N, int K, int num_gpus, int gpu_id,
                              cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                              const GemmKernel& kernel) {
    int N_local = N / num_gpus;

    // dW_i = X^T @ dY_i  [K, N/p]
    grad_gemm_at_b(d_X, d_dY_local, d_dW_shard, K, N_local, M,
                   kernel, handle, stream);

    // dX_partial = dY_i @ W_i^T  [M, K]
    grad_gemm_a_bt(d_dY_local, d_W_shard, d_dX_partial, M, K, N_local,
                   kernel, handle, d_W_shard_T, stream);

    // AllReduce: dX = sum(dX_partial)
    if (num_gpus == 1) {
        if (d_dX != d_dX_partial) {
            CUDA_CHECK(cudaMemcpyAsync(d_dX, d_dX_partial, M * K * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        NCCL_CHECK(ncclAllReduce(d_dX_partial, d_dX, M * K, ncclFloat, ncclSum, comm, stream));
    }
}

void row_parallel_backward(const float* d_X_shard, const float* d_W_shard, const float* d_dY,
                           float* d_dW_shard, float* d_dX_shard, float* d_W_shard_T, int M, int N,
                           int K, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
                           cudaStream_t stream, const GemmKernel& kernel) {
    int K_local = K / num_gpus;

    // dW_i = X_i^T @ dY  [K/p, N]
    grad_gemm_at_b(d_X_shard, d_dY, d_dW_shard, K_local, N, M,
                   kernel, handle, stream);

    // dX_i = dY @ W_i^T  [M, K/p]
    grad_gemm_a_bt(d_dY, d_W_shard, d_dX_shard, M, K_local, N,
                   kernel, handle, d_W_shard_T, stream);
}

// ============================================================================
// PARALLEL MLP BLOCK
// ============================================================================

void parallel_mlp_forward(const float* d_X, const float* d_W1_shard, const float* d_W2_shard,
                          float* d_hidden, float* d_Y_partial, float* d_Y, int M, int K, int H,
                          int N, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
                          cudaStream_t stream, const GemmKernel& kernel) {
    int H_local = H / num_gpus;

    local_gemm(d_X, d_W1_shard, d_hidden, M, H_local, K, kernel, stream);
    local_gemm(d_hidden, d_W2_shard, d_Y_partial, M, N, H_local, kernel, stream);

    if (num_gpus == 1) {
        if (d_Y != d_Y_partial) {
            CUDA_CHECK(cudaMemcpyAsync(d_Y, d_Y_partial, M * N * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        NCCL_CHECK(ncclAllReduce(d_Y_partial, d_Y, M * N, ncclFloat, ncclSum, comm, stream));
    }
}

void parallel_mlp_backward(const float* d_X, const float* d_W1_shard, const float* d_W2_shard,
                           const float* d_hidden, const float* d_dY, float* d_dW1_shard,
                           float* d_dW2_shard, float* d_d_hidden, float* d_dX_partial, float* d_dX,
                           int M, int K, int H, int N, int num_gpus, int gpu_id,
                           cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                           const GemmKernel& kernel) {
    int H_local = H / num_gpus;

    // Row backward (W2): dW2_i = hidden_i^T @ dY, d_hidden_i = dY @ W2_i^T
    grad_gemm_at_b(d_hidden, d_dY, d_dW2_shard, H_local, N, M,
                   kernel, handle, stream);

    {
        CudaMemory<float> d_W2_T(N * H_local);
        grad_gemm_a_bt(d_dY, d_W2_shard, d_d_hidden, M, H_local, N,
                       kernel, handle, d_W2_T.get(), stream);
    }

    // Column backward (W1): dW1_i = X^T @ d_hidden_i, dX_partial = d_hidden_i @ W1_i^T
    grad_gemm_at_b(d_X, d_d_hidden, d_dW1_shard, K, H_local, M,
                   kernel, handle, stream);

    {
        CudaMemory<float> d_W1_T(H_local * K);
        grad_gemm_a_bt(d_d_hidden, d_W1_shard, d_dX_partial, M, K, H_local,
                       kernel, handle, d_W1_T.get(), stream);
    }

    // AllReduce: dX = sum(dX_partial)
    if (num_gpus == 1) {
        if (d_dX != d_dX_partial) {
            CUDA_CHECK(cudaMemcpyAsync(d_dX, d_dX_partial, M * K * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        NCCL_CHECK(ncclAllReduce(d_dX_partial, d_dX, M * K, ncclFloat, ncclSum, comm, stream));
    }
}

// ============================================================================
// COMMUNICATION-COMPUTE OVERLAP
// ============================================================================

void row_parallel_forward_overlap(const float* d_X_shard, const float* d_W_shard, float* d_Y_local,
                                  float* d_Y_reduced, int M, int N, int K, int num_gpus, int gpu_id,
                                  int num_chunks, cublasHandle_t handle, ncclComm_t comm,
                                  cudaStream_t compute_stream, cudaStream_t comm_stream,
                                  const GemmKernel& kernel) {
    int K_local = K / num_gpus;
    int chunk_rows = (M + num_chunks - 1) / num_chunks;

    CudaEvent compute_done;

    for (int c = 0; c < num_chunks; c++) {
        int row_start = c * chunk_rows;
        int rows_this_chunk = (row_start + chunk_rows <= M) ? chunk_rows : (M - row_start);
        if (rows_this_chunk <= 0) break;

        const float* X_chunk = d_X_shard + row_start * K_local;
        float* Y_chunk = d_Y_local + row_start * N;
        float* Y_red_chunk = d_Y_reduced + row_start * N;

        kernel.launch(X_chunk, d_W_shard, Y_chunk, rows_this_chunk, N, K_local, compute_stream);

        compute_done.record(compute_stream);
        CUDA_CHECK(cudaStreamWaitEvent(comm_stream, compute_done, 0));

        if (num_gpus == 1) {
            if (Y_red_chunk != Y_chunk) {
                CUDA_CHECK(cudaMemcpyAsync(Y_red_chunk, Y_chunk,
                                           rows_this_chunk * N * sizeof(float),
                                           cudaMemcpyDeviceToDevice, comm_stream));
            }
        } else {
            NCCL_CHECK(ncclAllReduce(Y_chunk, Y_red_chunk, rows_this_chunk * N, ncclFloat, ncclSum,
                                     comm, comm_stream));
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
}
