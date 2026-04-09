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
 *   Used for: first linear layer in MLP block
 *
 * Row Parallelism:
 *   Weight W is split row-wise: W = [W_0; W_1; ...; W_{p-1}]
 *   Input X is split column-wise: X = [X_0 | X_1 | ... | X_{p-1}]
 *   Each GPU i computes: Y_i = X_i @ W_i  (partial sum)
 *   To get full output: AllReduce(Y_0 + Y_1 + ... + Y_{p-1})
 *   Used for: second linear layer in MLP block
 *
 * Together, column -> row parallelism forms a complete MLP block
 * with only ONE AllReduce between the two layers (forward pass)
 * and ONE AllReduce in the backward pass.
 *
 * Backward passes follow Megatron-LM:
 *   Column backward: dX = AllReduce(dY_i @ W_i^T), dW_i = X^T @ dY_i
 *   Row backward:    dX_i = dY @ W_i^T (no comm), dW_i = X_i^T @ dY
 */

#include "tensor_parallel.cuh"

#include "../utils/cuda_raii.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/nccl_utils.cuh"

// ============================================================================
// Helper: transpose matrix on GPU via cuBLAS (out-of-place)
// B[N x M] = A[M x N]^T
// ============================================================================
static void gpu_transpose(cublasHandle_t handle, const float* A, float* B, int M, int N,
                          cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    // cublasSgeam: B = alpha * A^T + beta * B
    CUBLAS_CHECK(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, M,
                             N,             // output is N x M in col-major = M x N row-major
                             &alpha, A, N,  // A is M x N row-major -> N x M col-major, lda=N
                             &beta, B, M,   // B placeholder, ldb=M
                             B, M));        // C output, ldc=M
}

// ============================================================================
// Helper: local GEMM with kernel selection
// C[M x N] = A[M x K] @ B[K x N]
// ============================================================================
static void local_gemm(const float* A, const float* B, float* C, int M, int N, int K,
                       GemmKernel kernel, cublasHandle_t handle, cudaStream_t stream) {
    dispatch_gemm_on_stream(kernel, A, B, C, M, N, K, handle, stream);
}

// ============================================================================
// FORWARD PASSES
// ============================================================================

void column_parallel_forward(const float* d_X, const float* d_W_shard, float* d_Y_local,
                             float* d_Y_full, int M, int N, int K, int num_gpus, int gpu_id,
                             cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                             GemmKernel kernel) {
    int N_local = N / num_gpus;

    // Local GEMM: Y_local = X @ W_shard
    local_gemm(d_X, d_W_shard, d_Y_local, M, N_local, K, kernel, handle, stream);

    // AllGather: collect all shards into d_Y_full
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
                          GemmKernel kernel) {
    int K_local = K / num_gpus;

    // Local GEMM: Y_local = X_shard @ W_shard
    local_gemm(d_X_shard, d_W_shard, d_Y_local, M, N, K_local, kernel, handle, stream);

    // AllReduce: sum partial results across GPUs
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
                              GemmKernel kernel) {
    int N_local = N / num_gpus;

    // dW_i = X^T @ dY_i  [K, M] x [M, N/p] = [K, N/p]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N_local, K, M, &alpha,
                                 d_dY_local, N_local, d_X, K, &beta, d_dW_shard, N_local));
    } else {
        CudaMemory<float> d_X_T(K * M);
        gpu_transpose(handle, d_X, d_X_T.get(), M, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_X_T.get(), d_dY_local, d_dW_shard, K, N_local, M);
    }

    // dX_partial = dY_i @ W_i^T  [M, N/p] x [N/p, K] = [M, K]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N_local, &alpha, d_W_shard,
                                 N_local, d_dY_local, N_local, &beta, d_dX_partial, K));
    } else {
        gpu_transpose(handle, d_W_shard, d_W_shard_T, K, N_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_dY_local, d_W_shard_T, d_dX_partial, M, K, N_local);
    }

    // AllReduce: dX = sum(dX_partial) across GPUs
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
                           cudaStream_t stream, GemmKernel kernel) {
    int K_local = K / num_gpus;

    // dW_i = X_i^T @ dY  [K/p, M] x [M, N] = [K/p, N]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K_local, M, &alpha, d_dY, N,
                                 d_X_shard, K_local, &beta, d_dW_shard, N));
    } else {
        CudaMemory<float> d_X_T(K_local * M);
        gpu_transpose(handle, d_X_shard, d_X_T.get(), M, K_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_X_T.get(), d_dY, d_dW_shard, K_local, N, M);
    }

    // dX_i = dY @ W_i^T  [M, N] x [N, K/p] = [M, K/p]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K_local, M, N, &alpha, d_W_shard,
                                 N, d_dY, N, &beta, d_dX_shard, K_local));
    } else {
        gpu_transpose(handle, d_W_shard, d_W_shard_T, K_local, N, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_dY, d_W_shard_T, d_dX_shard, M, K_local, N);
    }
}

// ============================================================================
// PARALLEL MLP BLOCK (FORWARD + BACKWARD)
// ============================================================================

void parallel_mlp_forward(const float* d_X, const float* d_W1_shard, const float* d_W2_shard,
                          float* d_hidden, float* d_Y_partial, float* d_Y, int M, int K, int H,
                          int N, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
                          cudaStream_t stream, GemmKernel kernel) {
    int H_local = H / num_gpus;

    // Layer 1: Column parallel -- hidden = X @ W1_shard  [M, H/p]
    local_gemm(d_X, d_W1_shard, d_hidden, M, H_local, K, kernel, handle, stream);

    // ReLU activation (in-place)
    // For simplicity, skip activation -- focus is on GEMM + communication

    // Layer 2: Row parallel -- Y_partial = hidden @ W2_shard  [M, N]
    local_gemm(d_hidden, d_W2_shard, d_Y_partial, M, N, H_local, kernel, handle, stream);

    // AllReduce: sum partial outputs
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
                           GemmKernel kernel) {
    int H_local = H / num_gpus;

    // --- Step 1: Row parallel backward (W2 layer) ---
    // dW2_i = hidden_i^T @ dY  [H/p, N]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, H_local, M, &alpha, d_dY, N,
                                 d_hidden, H_local, &beta, d_dW2_shard, N));
    } else {
        CudaMemory<float> d_hidden_T(H_local * M);
        gpu_transpose(handle, d_hidden, d_hidden_T.get(), M, H_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_hidden_T.get(), d_dY, d_dW2_shard, H_local, N, M);
    }

    // d_hidden_i = dY @ W2_i^T  [M, H/p]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H_local, M, N, &alpha,
                                 d_W2_shard, N, d_dY, N, &beta, d_d_hidden, H_local));
    } else {
        CudaMemory<float> d_W2_T(N * H_local);
        gpu_transpose(handle, d_W2_shard, d_W2_T.get(), H_local, N, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_dY, d_W2_T.get(), d_d_hidden, M, H_local, N);
    }

    // (ReLU backward would multiply d_hidden by mask here)

    // --- Step 2: Column parallel backward (W1 layer) ---
    // dW1_i = X^T @ d_hidden_i  [K, H/p]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, H_local, K, M, &alpha,
                                 d_d_hidden, H_local, d_X, K, &beta, d_dW1_shard, H_local));
    } else {
        CudaMemory<float> d_X_T(K * M);
        gpu_transpose(handle, d_X, d_X_T.get(), M, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_X_T.get(), d_d_hidden, d_dW1_shard, K, H_local, M);
    }

    // dX_partial = d_hidden_i @ W1_i^T  [M, K]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, H_local, &alpha,
                                 d_W1_shard, H_local, d_d_hidden, H_local, &beta, d_dX_partial, K));
    } else {
        CudaMemory<float> d_W1_T(H_local * K);
        gpu_transpose(handle, d_W1_shard, d_W1_T.get(), K, H_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto fn = kGemmLaunchFns[static_cast<int>(kernel)];
        fn(d_d_hidden, d_W1_T.get(), d_dX_partial, M, K, H_local);
    }

    // AllReduce: dX = sum(dX_partial) across GPUs
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
                                  GemmKernel kernel) {
    int K_local = K / num_gpus;
    int chunk_rows = (M + num_chunks - 1) / num_chunks;

    CudaEvent compute_done;

    for (int c = 0; c < num_chunks; c++) {
        int row_start = c * chunk_rows;
        int rows_this_chunk = (row_start + chunk_rows <= M) ? chunk_rows : (M - row_start);
        if (rows_this_chunk <= 0) break;

        // Pointers for this chunk
        const float* X_chunk = d_X_shard + row_start * K_local;
        float* Y_chunk = d_Y_local + row_start * N;
        float* Y_red_chunk = d_Y_reduced + row_start * N;

        // GEMM for this chunk on compute_stream
        dispatch_gemm_on_stream(kernel, X_chunk, d_W_shard, Y_chunk, rows_this_chunk, N, K_local,
                                handle, compute_stream);

        // Record event: GEMM for this chunk done
        compute_done.record(compute_stream);

        // Comm stream waits for compute to finish this chunk
        CUDA_CHECK(cudaStreamWaitEvent(comm_stream, compute_done, 0));

        // AllReduce this chunk on comm_stream
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

    // Wait for all communication to complete
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
}
