/*
 * Kernel 6: Vectorized Memory Access
 * ===================================
 * GPU memory transactions are most efficient at 128 bits (float4 = 4 floats).
 * By using float4 loads/stores, we:
 *   1. Reduce the number of memory instructions by 4x
 *   2. Better saturate memory bandwidth
 *   3. Allow the compiler to use LDG.128 instructions
 *
 * We also transpose A into shared memory (store As[k][m] instead of As[m][k])
 * to enable conflict-free access patterns during the compute phase.
 *
 * This kernel builds on the 2D block tiling structure from kernel 5.
 */

#include <cuda_runtime.h>

#define BM6 128
#define BN6 128
#define BK6 8
#define TM6 8
#define TN6 8

__global__ void gemm_vectorized(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
    // Transpose A in shared memory to avoid bank conflicts
    __shared__ float As[BK6][BM6];  // Transposed: [k][m]
    __shared__ float Bs[BK6][BN6];

    const int threads_per_block = (BM6 / TM6) * (BN6 / TN6);  // 256
    const int thread_col = threadIdx.x % (BN6 / TN6);
    const int thread_row = threadIdx.x / (BN6 / TN6);
    const int tid = threadIdx.x;

    const int block_row = blockIdx.y * BM6;
    const int block_col = blockIdx.x * BN6;

    float accum[TM6][TN6] = {{0.0f}};
    float a_cache[TM6];
    float b_cache[TN6];

    // Precompute load positions for A and B using float4
    // A tile: BM6 * BK6 = 1024 floats = 256 float4s → 1 float4 per thread
    const int a_load_row = tid / (BK6 / 4);  // row in A tile
    const int a_load_col = tid % (BK6 / 4);  // float4 column
    // B tile: BK6 * BN6 = 1024 floats = 256 float4s → 1 float4 per thread
    const int b_load_row = tid / (BN6 / 4);
    const int b_load_col = tid % (BN6 / 4);

    for (int t = 0; t < K; t += BK6) {
        // --- Vectorized load of A, stored transposed in smem ---
        {
            int gr = block_row + a_load_row;
            int gc = t + a_load_col * 4;
            if (gr < M && gc + 3 < K) {
                // Use float4 load from global memory
                float4 a4 = reinterpret_cast<const float4*>(&A[gr * K + gc])[0];
                // Store transposed: As[k][m]
                As[a_load_col * 4 + 0][a_load_row] = a4.x;
                As[a_load_col * 4 + 1][a_load_row] = a4.y;
                As[a_load_col * 4 + 2][a_load_row] = a4.z;
                As[a_load_col * 4 + 3][a_load_row] = a4.w;
            } else {
                // Scalar fallback for boundary
                for (int i = 0; i < 4; i++) {
                    int gci = gc + i;
                    As[a_load_col * 4 + i][a_load_row] =
                        (gr < M && gci < K) ? A[gr * K + gci] : 0.0f;
                }
            }
        }

        // --- Vectorized load of B ---
        {
            int gr = t + b_load_row;
            int gc = block_col + b_load_col * 4;
            if (gr < K && gc + 3 < N) {
                float4 b4 = reinterpret_cast<const float4*>(&B[gr * N + gc])[0];
                Bs[b_load_row][b_load_col * 4 + 0] = b4.x;
                Bs[b_load_row][b_load_col * 4 + 1] = b4.y;
                Bs[b_load_row][b_load_col * 4 + 2] = b4.z;
                Bs[b_load_row][b_load_col * 4 + 3] = b4.w;
            } else {
                for (int i = 0; i < 4; i++) {
                    int gci = gc + i;
                    Bs[b_load_row][b_load_col * 4 + i] =
                        (gr < K && gci < N) ? B[gr * N + gci] : 0.0f;
                }
            }
        }

        __syncthreads();

// --- Compute (same as kernel 5, but A is transposed in smem) ---
#pragma unroll
        for (int k = 0; k < BK6; k++) {
#pragma unroll
            for (int m = 0; m < TM6; m++) {
                a_cache[m] = As[k][thread_row * TM6 + m];  // Transposed access
            }
#pragma unroll
            for (int n = 0; n < TN6; n++) {
                b_cache[n] = Bs[k][thread_col * TN6 + n];
            }
#pragma unroll
            for (int m = 0; m < TM6; m++) {
#pragma unroll
                for (int n = 0; n < TN6; n++) {
                    accum[m][n] += a_cache[m] * b_cache[n];
                }
            }
        }

        __syncthreads();
    }

// --- Vectorized write of results using float4 ---
#pragma unroll
    for (int m = 0; m < TM6; m++) {
        int gr = block_row + thread_row * TM6 + m;
        if (gr < M) {
// Write TN6=8 elements as 2 float4s
#pragma unroll
            for (int n = 0; n < TN6; n += 4) {
                int gc = block_col + thread_col * TN6 + n;
                if (gc + 3 < N) {
                    float4 out;
                    out.x = accum[m][n + 0];
                    out.y = accum[m][n + 1];
                    out.z = accum[m][n + 2];
                    out.w = accum[m][n + 3];
                    reinterpret_cast<float4*>(&C[gr * N + gc])[0] = out;
                } else {
                    for (int i = 0; i < 4; i++) {
                        int gci = gc + i;
                        if (gci < N) C[gr * N + gci] = accum[m][n + i];
                    }
                }
            }
        }
    }
}

void launch_gemm_vectorized_stream(const float* A, const float* B, float* C, int M, int N, int K,
                                   cudaStream_t stream) {
    const int threads = (BM6 / TM6) * (BN6 / TN6);
    dim3 block(threads);
    dim3 grid((N + BN6 - 1) / BN6, (M + BM6 - 1) / BM6);
    gemm_vectorized<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
    launch_gemm_vectorized_stream(A, B, C, M, N, K, 0);
}
