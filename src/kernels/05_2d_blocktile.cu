/*
 * Kernel 5: 2D Block Tiling (Register Blocking)
 * ==============================================
 * Extends 1D tiling: each thread now computes a TM × TN sub-matrix of C,
 * stored entirely in registers. A single shared memory value is reused
 * TM times (from A) and TN times (from B).
 *
 * Arithmetic intensity per shared memory load:
 *   Kernel 3: 1 FMA per smem load
 *   Kernel 4: TM FMAs per smem load of B
 *   Kernel 5: TM*TN FMAs per pair of smem loads = TM+TN → 8*8/(8+8) = 4x
 *
 * Block config:
 *   BM=128, BN=128, BK=8, TM=8, TN=8
 *   Threads per block: (BN/TN) * (BM/TM) = 16 * 16 = 256
 *   Each thread: 8×8 = 64 output elements
 *   Registers per thread: 64 accumulators + 8+8 cached values = 80 regs
 *
 * This is the kernel structure used by high-performance GEMM implementations.
 */

#include <cuda_runtime.h>

#define BM5 128
#define BN5 128
#define BK5 8
#define TM5 8
#define TN5 8

__global__ void gemm_2d_blocktile(const float * __restrict__ A,
                                   const float * __restrict__ B,
                                   float * __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[BM5][BK5];
    __shared__ float Bs[BK5][BN5];

    // Thread position within the block's thread grid
    // Block has (BN/TN) × (BM/TM) = 16×16 = 256 threads
    const int thread_col = threadIdx.x % (BN5 / TN5);  // 0..15
    const int thread_row = threadIdx.x / (BN5 / TN5);  // 0..15

    // Linear thread ID for cooperative loading
    const int tid = threadIdx.x;
    const int num_threads = (BM5 / TM5) * (BN5 / TN5);  // 256

    // Accumulator: TM×TN register tile
    float accum[TM5][TN5] = {{0.0f}};

    // Register caches for the current k-step
    float a_cache[TM5];
    float b_cache[TN5];

    // Block base positions
    const int block_row = blockIdx.y * BM5;
    const int block_col = blockIdx.x * BN5;

    // How many elements each thread loads cooperatively
    // As: BM5 * BK5 = 128*8 = 1024, with 256 threads → 4 per thread
    // Bs: BK5 * BN5 = 8*128 = 1024, with 256 threads → 4 per thread
    const int a_loads = (BM5 * BK5) / num_threads;  // 4
    const int b_loads = (BK5 * BN5) / num_threads;  // 4

    for (int t = 0; t < K; t += BK5) {
        // --- Load A tile ---
        #pragma unroll
        for (int i = 0; i < a_loads; i++) {
            int idx = tid + i * num_threads;
            int lr = idx / BK5;
            int lc = idx % BK5;
            int gr = block_row + lr;
            int gc = t + lc;
            As[lr][lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        // --- Load B tile ---
        #pragma unroll
        for (int i = 0; i < b_loads; i++) {
            int idx = tid + i * num_threads;
            int lr = idx / BN5;
            int lc = idx % BN5;
            int gr = t + lr;
            int gc = block_col + lc;
            Bs[lr][lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // --- Compute TM×TN outer product for each k ---
        #pragma unroll
        for (int k = 0; k < BK5; k++) {
            // Load column of A tile into registers
            #pragma unroll
            for (int m = 0; m < TM5; m++) {
                a_cache[m] = As[thread_row * TM5 + m][k];
            }
            // Load row of B tile into registers
            #pragma unroll
            for (int n = 0; n < TN5; n++) {
                b_cache[n] = Bs[k][thread_col * TN5 + n];
            }
            // Outer product
            #pragma unroll
            for (int m = 0; m < TM5; m++) {
                #pragma unroll
                for (int n = 0; n < TN5; n++) {
                    accum[m][n] += a_cache[m] * b_cache[n];
                }
            }
        }

        __syncthreads();
    }

    // --- Write results ---
    #pragma unroll
    for (int m = 0; m < TM5; m++) {
        int gr = block_row + thread_row * TM5 + m;
        #pragma unroll
        for (int n = 0; n < TN5; n++) {
            int gc = block_col + thread_col * TN5 + n;
            if (gr < M && gc < N) {
                C[gr * N + gc] = accum[m][n];
            }
        }
    }
}

void launch_gemm_2d_blocktile_stream(const float *A, const float *B, float *C,
                                     int M, int N, int K, cudaStream_t stream) {
    const int threads = (BM5 / TM5) * (BN5 / TN5);  // 256
    dim3 block(threads);
    dim3 grid((N + BN5 - 1) / BN5, (M + BM5 - 1) / BM5);
    gemm_2d_blocktile<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_2d_blocktile(const float *A, const float *B, float *C,
                              int M, int N, int K) {
    launch_gemm_2d_blocktile_stream(A, B, C, M, N, K, 0);
}
