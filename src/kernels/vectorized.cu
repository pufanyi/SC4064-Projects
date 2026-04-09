/*
 * Vectorized GEMM Kernel
 * ======================
 * Uses float4 (128-bit) loads and stores A transposed in shared memory
 * to eliminate bank conflicts. Builds on the 2D block tiling structure.
 */

#include <cuda_runtime.h>

#include "kernel_registry.cuh"

namespace {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 8;
constexpr int TN = 8;

__global__ void gemm_vectorized(const float* __restrict__ A, const float* __restrict__ B,
                                float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BK][BM];  // Transposed: [k][m]
    __shared__ float Bs[BK][BN];

    const int thread_col = threadIdx.x % (BN / TN);
    const int thread_row = threadIdx.x / (BN / TN);
    const int tid = threadIdx.x;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float accum[TM][TN] = {{0.0f}};
    float a_cache[TM];
    float b_cache[TN];

    const int a_load_row = tid / (BK / 4);
    const int a_load_col = tid % (BK / 4);
    const int b_load_row = tid / (BN / 4);
    const int b_load_col = tid % (BN / 4);

    for (int t = 0; t < K; t += BK) {
        // Vectorized load of A, stored transposed in smem
        {
            int gr = block_row + a_load_row;
            int gc = t + a_load_col * 4;
            if (gr < M && gc + 3 < K) {
                float4 a4 = reinterpret_cast<const float4*>(&A[gr * K + gc])[0];
                As[a_load_col * 4 + 0][a_load_row] = a4.x;
                As[a_load_col * 4 + 1][a_load_row] = a4.y;
                As[a_load_col * 4 + 2][a_load_row] = a4.z;
                As[a_load_col * 4 + 3][a_load_row] = a4.w;
            } else {
                for (int i = 0; i < 4; i++) {
                    int gci = gc + i;
                    As[a_load_col * 4 + i][a_load_row] =
                        (gr < M && gci < K) ? A[gr * K + gci] : 0.0f;
                }
            }
        }

        // Vectorized load of B
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

#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int m = 0; m < TM; m++) {
                a_cache[m] = As[k][thread_row * TM + m];
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

// Vectorized write
#pragma unroll
    for (int m = 0; m < TM; m++) {
        int gr = block_row + thread_row * TM + m;
        if (gr < M) {
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                int gc = block_col + thread_col * TN + n;
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

class VectorizedKernel : public GemmKernel {
   public:
    const char* name() const override { return "Vectorized"; }

    void launch(const float* A, const float* B, float* C, int M, int N, int K,
                cudaStream_t stream) const override {
        constexpr int threads = (BM / TM) * (BN / TN);
        dim3 block(threads);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        gemm_vectorized<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
    }
};

static int reg = KernelRegistry::add(std::make_unique<VectorizedKernel>());

}  // namespace
