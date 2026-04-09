/*
 * Warp Tiling GEMM Kernel
 * =======================
 * Hierarchical tiling: Block(128×128) → Warp(32×64) → Thread(8×8).
 * Improves shared memory locality within each warp and enables
 * better instruction-level parallelism.
 *
 * Warps per block: (128/32)*(128/64) = 4*2 = 8
 * Threads per warp: (32/8)*(64/8) = 4*8 = 32 ✓
 * Total: 256 threads
 */

#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int WM = 32;
constexpr int WN = 64;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void gemm_warptile(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BK][BM];  // Transposed
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int warps_per_row = BN / WN;
    const int warp_row = warp_id / warps_per_row;
    const int warp_col = warp_id % warps_per_row;

    const int thread_col_in_warp = lane_id % (WN / TN);
    const int thread_row_in_warp = lane_id / (WN / TN);

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float accum[TM][TN] = {{0.0f}};
    float a_cache[TM];
    float b_cache[TN];

    constexpr int num_threads = 256;

    for (int t = 0; t < K; t += BK) {
#pragma unroll
        for (int i = 0; i < (BM * BK) / num_threads; i++) {
            int idx = tid + i * num_threads;
            int lr = idx / BK;
            int lc = idx % BK;
            int gr = block_row + lr;
            int gc = t + lc;
            As[lc][lr] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
#pragma unroll
        for (int i = 0; i < (BK * BN) / num_threads; i++) {
            int idx = tid + i * num_threads;
            int lr = idx / BN;
            int lc = idx % BN;
            int gr = t + lr;
            int gc = block_col + lc;
            Bs[lr][lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
                int sm_row = warp_row * WM + thread_row_in_warp * TM + m;
                a_cache[m] = As[k][sm_row];
            }
#pragma unroll
            for (int n = 0; n < TN; n++) {
                int sm_col = warp_col * WN + thread_col_in_warp * TN + n;
                b_cache[n] = Bs[k][sm_col];
            }
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_cache[m] * b_cache[n];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; m++) {
        int gr = block_row + warp_row * WM + thread_row_in_warp * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n++) {
            int gc = block_col + warp_col * WN + thread_col_in_warp * TN + n;
            if (gr < M && gc < N) {
                C[gr * N + gc] = accum[m][n];
            }
        }
    }
}

class WarpTileKernel : public GemmKernel {
   public:
    const char* name() const override { return "WarpTile"; }

    void launch(const float* A, const float* B, float* C, int M, int N, int K,
                cudaStream_t stream) const override {
        constexpr int warps = (BM / WM) * (BN / WN);
        constexpr int threads = warps * 32;
        dim3 block(threads);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_warptile<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

static int reg = KernelRegistry::add(std::make_unique<WarpTileKernel>());

}  // namespace
