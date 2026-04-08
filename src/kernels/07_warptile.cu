/*
 * Kernel 7: Warp Tiling
 * =====================
 * Adds a warp-level tiling layer: Block → Warp → Thread.
 *
 * Motivation: In kernels 5-6, threads within a warp may access scattered
 * shared memory locations. By organizing warps into 2D sub-tiles of the
 * block tile, we can:
 *   1. Improve shared memory access locality within a warp
 *   2. Reduce the working set per warp → better L1/register utilization
 *   3. Enable more effective instruction-level parallelism
 *
 * Hierarchy:
 *   Block tile:  BM × BN = 128 × 128
 *   Warp tile:   WM × WN = 64 × 64   (each warp computes this sub-tile)
 *   Thread tile: TM × TN = 8 × 8     (each thread computes this)
 *   → Threads per warp used: (WM/TM) × (WN/TN) = 8×8 = 64... too many.
 *
 * Adjusted: WM=64, WN=32, TM=8, TN=4 → 8*8=64? No, (64/8)*(32/4)=8*8=64.
 * A warp has 32 threads, so: WM=32, WN=32, TM=8, TN=8 → (4)*(4)=16... no.
 *
 * Let's use: BM=128, BN=128, WM=64, WN=64, TM=8, TN=8
 * Threads per warp sub-tile: (WM/TM)*(WN/TN) = 8*8 = 64 → 2 warps per sub-tile
 * Or: WM=32, WN=64, TM=8, TN=8 → 4*8=32 threads per warp ✓
 * Warps per block: (BM/WM)*(BN/WN) = 4*2 = 8 warps = 256 threads ✓
 */

#include <cuda_runtime.h>

#define BM7 128
#define BN7 128
#define BK7 8
#define WM7 32     // Warp tile M
#define WN7 64     // Warp tile N
#define TM7 8      // Thread tile M
#define TN7 8      // Thread tile N

// Warps per block: (BM/WM) * (BN/WN) = 4 * 2 = 8
// Threads per warp: (WM/TM) * (WN/TN) = 4 * 8 = 32 ✓
// Total threads: 8 * 32 = 256

__global__ void gemm_warptile(const float * __restrict__ A,
                               const float * __restrict__ B,
                               float * __restrict__ C,
                               int M, int N, int K) {
    __shared__ float As[BK7][BM7];   // Transposed
    __shared__ float Bs[BK7][BN7];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Warp position within block tile
    const int warps_per_row = BN7 / WN7;  // 2
    const int warp_row = warp_id / warps_per_row;  // 0..3
    const int warp_col = warp_id % warps_per_row;  // 0..1

    // Thread position within warp tile
    const int thread_col_in_warp = lane_id % (WN7 / TN7);  // 0..7
    const int thread_row_in_warp = lane_id / (WN7 / TN7);  // 0..3

    const int block_row = blockIdx.y * BM7;
    const int block_col = blockIdx.x * BN7;

    float accum[TM7][TN7] = {{0.0f}};
    float a_cache[TM7];
    float b_cache[TN7];

    const int num_threads = (BM7 / TM7) * (BN7 / TN7);  // 256 (but via warps)

    for (int t = 0; t < K; t += BK7) {
        // --- Cooperative load (same as kernel 6) ---
        // A: BM7*BK7 = 1024 floats, 256 threads → 4 per thread
        #pragma unroll
        for (int i = 0; i < (BM7 * BK7) / num_threads; i++) {
            int idx = tid + i * num_threads;
            int lr = idx / BK7;
            int lc = idx % BK7;
            int gr = block_row + lr;
            int gc = t + lc;
            As[lc][lr] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        // B: BK7*BN7 = 1024 floats
        #pragma unroll
        for (int i = 0; i < (BK7 * BN7) / num_threads; i++) {
            int idx = tid + i * num_threads;
            int lr = idx / BN7;
            int lc = idx % BN7;
            int gr = t + lr;
            int gc = block_col + lc;
            Bs[lr][lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        // --- Compute using warp-level tiling ---
        #pragma unroll
        for (int k = 0; k < BK7; k++) {
            // Load from warp's region of shared memory
            #pragma unroll
            for (int m = 0; m < TM7; m++) {
                int sm_row = warp_row * WM7 + thread_row_in_warp * TM7 + m;
                a_cache[m] = As[k][sm_row];
            }
            #pragma unroll
            for (int n = 0; n < TN7; n++) {
                int sm_col = warp_col * WN7 + thread_col_in_warp * TN7 + n;
                b_cache[n] = Bs[k][sm_col];
            }
            #pragma unroll
            for (int m = 0; m < TM7; m++) {
                #pragma unroll
                for (int n = 0; n < TN7; n++) {
                    accum[m][n] += a_cache[m] * b_cache[n];
                }
            }
        }

        __syncthreads();
    }

    // --- Write results ---
    #pragma unroll
    for (int m = 0; m < TM7; m++) {
        int gr = block_row + warp_row * WM7 + thread_row_in_warp * TM7 + m;
        #pragma unroll
        for (int n = 0; n < TN7; n++) {
            int gc = block_col + warp_col * WN7 + thread_col_in_warp * TN7 + n;
            if (gr < M && gc < N) {
                C[gr * N + gc] = accum[m][n];
            }
        }
    }
}

void launch_gemm_warptile_stream(const float *A, const float *B, float *C,
                                 int M, int N, int K, cudaStream_t stream) {
    const int warps = (BM7 / WM7) * (BN7 / WN7);       // 8
    const int threads = warps * 32;                       // 256
    dim3 block(threads);
    dim3 grid((N + BN7 - 1) / BN7, (M + BM7 - 1) / BM7);
    gemm_warptile<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_warptile(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    launch_gemm_warptile_stream(A, B, C, M, N, K, 0);
}
