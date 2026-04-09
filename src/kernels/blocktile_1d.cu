/*
 * 1D Block Tiling GEMM Kernel (Thread Coarsening along M)
 * ========================================================
 * Each thread computes TM=8 rows. A single B shared memory load
 * is reused across TM accumulations.
 * Block: (64, 8) = 512 threads.
 */

#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 8;
constexpr int TM = 8;

__global__ void gemm_blocktile_1d(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_base = by * BM + ty * TM;
    int col = bx * BN + tx;

    float accum[TM] = {0.0f};

    int tid = ty * BN + tx;

    for (int t = 0; t < K; t += BK) {
        {
            int load_row = tid / BK;
            int load_col = tid % BK;
            int g_row = by * BM + load_row;
            int g_col = t + load_col;
            As[load_row][load_col] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }
        {
            int load_row = tid / BN;
            int load_col = tid % BN;
            int g_row = t + load_row;
            int g_col = bx * BN + load_col;
            Bs[load_row][load_col] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k++) {
            float b_val = Bs[k][tx];
#pragma unroll
            for (int m = 0; m < TM; m++) {
                accum[m] += As[ty * TM + m][k] * b_val;
            }
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; m++) {
        int g_row = row_base + m;
        if (g_row < M && col < N) {
            C[g_row * N + col] = accum[m];
        }
    }
}

class BlockTile1DKernel : public GemmKernel {
   public:
    const char* name() const override { return "BlockTile1D"; }

    void launch(const float* A, const float* B, float* C,
                int M, int N, int K, cudaStream_t stream) const override {
        dim3 block(BN, BM / TM);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_blocktile_1d<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

static int reg = KernelRegistry::add(std::make_unique<BlockTile1DKernel>());

}  // namespace
