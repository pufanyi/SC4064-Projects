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

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>

#include "../kernels/gemm_dispatch.cuh"
#include "../kernels/kernels.cuh"
#include "../utils/cuda_utils.cuh"

#define NCCL_CHECK(cmd)                                                                            \
    do {                                                                                           \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess) {                                                                    \
            fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

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
                             &alpha, A, N,  // A is M x N row-major → N x M col-major, lda=N
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

// ---------------------------------------------------------------------------
// Column Parallel Linear Forward: Y = X @ W  where W is column-sharded
// ---------------------------------------------------------------------------
// Each GPU holds W_i of shape [K, N/p] and computes Y_i = X @ W_i
// Output: AllGather -> full Y of shape [M, N] (optional)
void column_parallel_forward(const float* d_X,        // [M, K] -- replicated on all GPUs
                             const float* d_W_shard,  // [K, N/p] -- local weight shard
                             float* d_Y_local,        // [M, N/p] -- local output
                             float* d_Y_full,  // [M, N] -- gathered output (optional, can be NULL)
                             int M, int N, int K, int num_gpus, int gpu_id, cublasHandle_t handle,
                             ncclComm_t comm, cudaStream_t stream,
                             GemmKernel kernel = GemmKernel::CUBLAS) {
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

// ---------------------------------------------------------------------------
// Row Parallel Linear Forward: Y = X @ W  where W is row-sharded
// ---------------------------------------------------------------------------
// Each GPU holds W_i of shape [K/p, N] and X_i of shape [M, K/p]
// Computes partial Y_i = X_i @ W_i, then AllReduce to sum
void row_parallel_forward(const float* d_X_shard,  // [M, K/p] -- local input shard
                          const float* d_W_shard,  // [K/p, N] -- local weight shard
                          float* d_Y_local,        // [M, N] -- local partial output
                          float* d_Y_reduced,      // [M, N] -- all-reduced output
                          int M, int N, int K, int num_gpus, int gpu_id, cublasHandle_t handle,
                          ncclComm_t comm, cudaStream_t stream,
                          GemmKernel kernel = GemmKernel::CUBLAS) {
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

// ---------------------------------------------------------------------------
// Column Parallel Linear Backward
// ---------------------------------------------------------------------------
// Forward was: Y_i = X @ W_i  (each GPU has W_i [K, N/p], full X [M, K])
//
// Backward:
//   dW_i = X^T @ dY_i        [K, N/p]  -- local, no communication
//   dX_i = dY_i @ W_i^T      [M, K]    -- partial gradient
//   dX   = AllReduce(dX_i)    [M, K]    -- sum partials across GPUs
void column_parallel_backward(
    const float* d_X,         // [M, K] -- replicated input (saved from forward)
    const float* d_W_shard,   // [K, N/p] -- local weight shard
    const float* d_dY_local,  // [M, N/p] -- incoming gradient (local shard)
    float* d_dW_shard,        // [K, N/p] -- weight gradient output
    float* d_dX_partial,      // [M, K] -- partial input gradient (workspace)
    float* d_dX,              // [M, K] -- full input gradient (after AllReduce)
    float* d_W_shard_T,       // [N/p, K] -- workspace for W^T
    int M, int N, int K, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
    cudaStream_t stream, GemmKernel kernel = GemmKernel::CUBLAS) {
    int N_local = N / num_gpus;

    // dW_i = X^T @ dY_i  [K, M] x [M, N/p] = [K, N/p]
    // Using cuBLAS for transpose: X^T @ dY_i
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        // C = A^T @ B  in row-major:
        // cublasSgemm(N, T, ...) with col-major trick
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N_local, K, M, &alpha,
                                 d_dY_local, N_local,           // dY^T in col-major
                                 d_X, K,                        // X^T in col-major
                                 &beta, d_dW_shard, N_local));  // dW^T in col-major
    } else {
        // Custom kernels expect row-major C = A @ B
        // Need explicit transpose: X_T [K, M] = transpose(X [M, K])
        // Then: dW = X_T @ dY_local
        float* d_X_T;
        CUDA_CHECK(cudaMalloc(&d_X_T, K * M * sizeof(float)));
        gpu_transpose(handle, d_X, d_X_T, M, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        fn(d_X_T, d_dY_local, d_dW_shard, K, N_local, M);

        CUDA_CHECK(cudaFree(d_X_T));
    }

    // dX_partial = dY_i @ W_i^T  [M, N/p] x [N/p, K] = [M, K]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, N_local, &alpha, d_W_shard,
                                 N_local, d_dY_local, N_local, &beta, d_dX_partial, K));
    } else {
        // Explicit transpose W^T
        gpu_transpose(handle, d_W_shard, d_W_shard_T, K, N_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
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

// ---------------------------------------------------------------------------
// Row Parallel Linear Backward
// ---------------------------------------------------------------------------
// Forward was: Y = AllReduce(X_i @ W_i)
//   Each GPU has W_i [K/p, N], X_i [M, K/p]
//
// Backward:
//   dW_i = X_i^T @ dY     [K/p, N]  -- local, no communication
//   dX_i = dY @ W_i^T     [M, K/p]  -- local gradient shard, no communication
//
// No AllReduce needed in backward! The gradient naturally splits because
// each GPU only needs its own shard dX_i.
void row_parallel_backward(
    const float* d_X_shard,  // [M, K/p] -- local input shard (saved from forward)
    const float* d_W_shard,  // [K/p, N] -- local weight shard
    const float* d_dY,       // [M, N] -- incoming gradient (replicated after AllReduce in fwd)
    float* d_dW_shard,       // [K/p, N] -- weight gradient output
    float* d_dX_shard,       // [M, K/p] -- input gradient shard output
    float* d_W_shard_T,      // [N, K/p] -- workspace for W^T
    int M, int N, int K, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
    cudaStream_t stream, GemmKernel kernel = GemmKernel::CUBLAS) {
    int K_local = K / num_gpus;

    // dW_i = X_i^T @ dY  [K/p, M] x [M, N] = [K/p, N]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, K_local, M, &alpha, d_dY, N,
                                 d_X_shard, K_local, &beta, d_dW_shard, N));
    } else {
        float* d_X_T;
        CUDA_CHECK(cudaMalloc(&d_X_T, K_local * M * sizeof(float)));
        gpu_transpose(handle, d_X_shard, d_X_T, M, K_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        fn(d_X_T, d_dY, d_dW_shard, K_local, N, M);

        CUDA_CHECK(cudaFree(d_X_T));
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

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        fn(d_dY, d_W_shard_T, d_dX_shard, M, K_local, N);
    }
}

// ============================================================================
// PARALLEL MLP BLOCK (FORWARD + BACKWARD)
// ============================================================================

// ---------------------------------------------------------------------------
// Parallel MLP Forward
// ---------------------------------------------------------------------------
// MLP(X) = ReLU(X @ W1) @ W2
// W1: column parallel [K, H] -> each GPU has [K, H/p]
// W2: row parallel    [H, N] -> each GPU has [H/p, N]
// Communication: ONE AllReduce (in row parallel of W2)
void parallel_mlp_forward(const float* d_X,         // [M, K] replicated
                          const float* d_W1_shard,  // [K, H/p] column shard
                          const float* d_W2_shard,  // [H/p, N] row shard
                          float* d_hidden,          // [M, H/p] intermediate
                          float* d_Y_partial,       // [M, N] partial output
                          float* d_Y,               // [M, N] final output
                          int M, int K, int H, int N, int num_gpus, int gpu_id,
                          cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                          GemmKernel kernel = GemmKernel::CUBLAS) {
    int H_local = H / num_gpus;

    // Layer 1: Column parallel -- hidden = X @ W1_shard  [M, H/p]
    local_gemm(d_X, d_W1_shard, d_hidden, M, H_local, K, kernel, handle, stream);

    // ReLU activation (in-place)
    // For simplicity, skip activation — focus is on GEMM + communication

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

// ---------------------------------------------------------------------------
// Parallel MLP Backward
// ---------------------------------------------------------------------------
// Forward: hidden_i = X @ W1_i,  Y = AllReduce(hidden_i @ W2_i)
//
// Backward (reverse order):
//   1. Row parallel backward (W2): no AllReduce needed
//      dW2_i = hidden_i^T @ dY    [H/p, N]
//      d_hidden_i = dY @ W2_i^T   [M, H/p]
//
//   2. (ReLU backward would go here if activation were applied)
//
//   3. Column parallel backward (W1): ONE AllReduce
//      dW1_i = X^T @ d_hidden_i          [K, H/p]
//      dX_partial = d_hidden_i @ W1_i^T   [M, K]
//      dX = AllReduce(dX_partial)          [M, K]
//
// Total communication in backward: ONE AllReduce (symmetric with forward)
void parallel_mlp_backward(const float* d_X,         // [M, K] saved from forward
                           const float* d_W1_shard,  // [K, H/p] saved from forward
                           const float* d_W2_shard,  // [H/p, N] saved from forward
                           const float* d_hidden,    // [M, H/p] saved from forward
                           const float* d_dY,        // [M, N] incoming gradient
                           float* d_dW1_shard,       // [K, H/p] output: W1 gradient
                           float* d_dW2_shard,       // [H/p, N] output: W2 gradient
                           float* d_d_hidden,        // [M, H/p] workspace: hidden gradient
                           float* d_dX_partial,      // [M, K] workspace: partial X gradient
                           float* d_dX,              // [M, K] output: full X gradient
                           int M, int K, int H, int N, int num_gpus, int gpu_id,
                           cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                           GemmKernel kernel = GemmKernel::CUBLAS) {
    int H_local = H / num_gpus;

    // --- Step 1: Row parallel backward (W2 layer) ---
    // dW2_i = hidden_i^T @ dY  [H/p, N]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, H_local, M, &alpha, d_dY, N,
                                 d_hidden, H_local, &beta, d_dW2_shard, N));
    } else {
        float* d_hidden_T;
        CUDA_CHECK(cudaMalloc(&d_hidden_T, H_local * M * sizeof(float)));
        gpu_transpose(handle, d_hidden, d_hidden_T, M, H_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        fn(d_hidden_T, d_dY, d_dW2_shard, H_local, N, M);
        CUDA_CHECK(cudaFree(d_hidden_T));
    }

    // d_hidden_i = dY @ W2_i^T  [M, K/p]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H_local, M, N, &alpha,
                                 d_W2_shard, N, d_dY, N, &beta, d_d_hidden, H_local));
    } else {
        float* d_W2_T;
        CUDA_CHECK(cudaMalloc(&d_W2_T, N * H_local * sizeof(float)));
        gpu_transpose(handle, d_W2_shard, d_W2_T, H_local, N, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        fn(d_dY, d_W2_T, d_d_hidden, M, H_local, N);
        CUDA_CHECK(cudaFree(d_W2_T));
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
        float* d_X_T;
        CUDA_CHECK(cudaMalloc(&d_X_T, K * M * sizeof(float)));
        gpu_transpose(handle, d_X, d_X_T, M, K, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        fn(d_X_T, d_d_hidden, d_dW1_shard, K, H_local, M);
        CUDA_CHECK(cudaFree(d_X_T));
    }

    // dX_partial = d_hidden_i @ W1_i^T  [M, K]
    if (kernel == GemmKernel::CUBLAS) {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, K, M, H_local, &alpha,
                                 d_W1_shard, H_local, d_d_hidden, H_local, &beta, d_dX_partial, K));
    } else {
        float* d_W1_T;
        CUDA_CHECK(cudaMalloc(&d_W1_T, H_local * K * sizeof(float)));
        gpu_transpose(handle, d_W1_shard, d_W1_T, K, H_local, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        GemmLaunchFn fn = get_gemm_launch_fn(kernel);
        fn(d_d_hidden, d_W1_T, d_dX_partial, M, K, H_local);
        CUDA_CHECK(cudaFree(d_W1_T));
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

// ---------------------------------------------------------------------------
// Row Parallel Forward with Overlap
// ---------------------------------------------------------------------------
// Splits the GEMM output into chunks. Each chunk's AllReduce overlaps with
// the next chunk's GEMM computation using separate CUDA streams.
//
// This is beneficial when comm and compute take similar time.
// Uses ncclGroupStart/End to batch NCCL operations.
void row_parallel_forward_overlap(const float* d_X_shard,  // [M, K/p]
                                  const float* d_W_shard,  // [K/p, N]
                                  float* d_Y_local,        // [M, N] partial output
                                  float* d_Y_reduced,      // [M, N] final output
                                  int M, int N, int K, int num_gpus, int gpu_id,
                                  int num_chunks,  // number of row chunks for overlap
                                  cublasHandle_t handle, ncclComm_t comm,
                                  cudaStream_t compute_stream, cudaStream_t comm_stream,
                                  GemmKernel kernel = GemmKernel::CUBLAS) {
    int K_local = K / num_gpus;
    int chunk_rows = (M + num_chunks - 1) / num_chunks;

    // Event for synchronization between streams
    cudaEvent_t compute_done;
    CUDA_CHECK(cudaEventCreate(&compute_done));

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
        CUDA_CHECK(cudaEventRecord(compute_done, compute_stream));

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

    CUDA_CHECK(cudaEventDestroy(compute_done));
}
