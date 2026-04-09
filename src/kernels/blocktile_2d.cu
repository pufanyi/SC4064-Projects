/*
 * 2D Block Tiling GEMM Kernel (Register Blocking)
 * ================================================
 * Each thread computes a TM×TN=8×8 sub-tile in registers.
 * BM=128, BN=128, BK=8, 256 threads per block.
 */

#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void gemm_blocktile_2d(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const int thread_col = threadIdx.x % (BN / TN);
    const int thread_row = threadIdx.x / (BN / TN);
    const int tid = threadIdx.x;
    const int num_threads = (BM / TM) * (BN / TN);

    float accum[TM][TN] = {{0.0f}};
    float a_cache[TM];
    float b_cache[TN];

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    const int a_loads = (BM * BK) / num_threads;
    const int b_loads = (BK * BN) / num_threads;

    for (int t = 0; t < K; t += BK) {
#pragma unroll
        for (int i = 0; i < a_loads; i++) {
            int idx = tid + i * num_threads;
            int lr = idx / BK;
            int lc = idx % BK;
            int gr = block_row + lr;
            int gc = t + lc;
            As[lr][lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

#pragma unroll
        for (int i = 0; i < b_loads; i++) {
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
                a_cache[m] = As[thread_row * TM + m][k];
            }
#pragma unroll
            for (int n = 0; n < TN; n++) {
                b_cache[n] = Bs[k][thread_col * TN + n];
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
        int gr = block_row + thread_row * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n++) {
            int gc = block_col + thread_col * TN + n;
            if (gr < M && gc < N) {
                C[gr * N + gc] = accum[m][n];
            }
        }
    }
}

class BlockTile2DKernel : public GemmKernel {
   public:
    const char* name() const override { return "BlockTile2D"; }

    void launch(const float* A, const float* B, float* C,
                int M, int N, int K, cudaStream_t stream) const override {
        constexpr int threads = (BM / TM) * (BN / TN);
        dim3 block(threads);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_blocktile_2d<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

static int reg = KernelRegistry::add(std::make_unique<BlockTile2DKernel>());

}  // namespace
