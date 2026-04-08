/*
 * Kernel 4: 1D Block Tiling (Thread Coarsening along M)
 * =====================================================
 * In kernel 3, each thread computes 1 output element and performs TILE_SIZE
 * FMAs per tile iteration → 32 FMAs. The bottleneck shifts to shared memory
 * load latency and synchronization overhead.
 *
 * Idea: Each thread computes TM elements along the M dimension. A single
 * load from Bs[k][threadIdx.x] is reused across TM accumulations → higher
 * arithmetic intensity per shared memory load.
 *
 * Block config: BN threads along N, BM/TM threads along M.
 *   - Block dimensions: (BN, BM/TM) = (64, 64/8) = (64, 8) = 512 threads
 *   - Each thread: TM=8 output elements
 *   - Shared memory: BM*BK + BK*BN floats
 *
 * This increases FMAs per thread from BK to BK*TM per tile, improving
 * the compute-to-memory ratio significantly.
 */

#include <cuda_runtime.h>

// Tile dimensions
#define BM4 64     // Block tile M
#define BN4 64     // Block tile N
#define BK4 8      // Tile along K per iteration
#define TM4 8      // Thread tile M: each thread computes 8 rows

__global__ void gemm_1d_blocktile(const float * __restrict__ A,
                                   const float * __restrict__ B,
                                   float * __restrict__ C,
                                   int M, int N, int K) {
    __shared__ float As[BM4][BK4];
    __shared__ float Bs[BK4][BN4];

    // Block position
    int bx = blockIdx.x;  // along N
    int by = blockIdx.y;  // along M

    // Thread position within block
    int tx = threadIdx.x;  // 0..BN-1 → column within block tile
    int ty = threadIdx.y;  // 0..BM/TM-1 → which group of TM rows

    // Global row/col base for this thread
    int row_base = by * BM4 + ty * TM4;
    int col = bx * BN4 + tx;

    // Accumulator registers: one per output row this thread owns
    float accum[TM4] = {0.0f};

    // Number of threads in block for cooperative loading
    int tid = ty * BN4 + tx;
    int num_threads = (BM4 / TM4) * BN4;  // 8 * 64 = 512

    // Slide along K
    for (int t = 0; t < K; t += BK4) {
        // --- Cooperative load of As[BM4][BK4] ---
        // Total elements: BM4 * BK4 = 64 * 8 = 512 → 1 element per thread
        {
            int load_row = tid / BK4;
            int load_col = tid % BK4;
            int g_row = by * BM4 + load_row;
            int g_col = t + load_col;
            As[load_row][load_col] =
                (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // --- Cooperative load of Bs[BK4][BN4] ---
        // Total elements: BK4 * BN4 = 8 * 64 = 512 → 1 element per thread
        {
            int load_row = tid / BN4;
            int load_col = tid % BN4;
            int g_row = t + load_row;
            int g_col = bx * BN4 + load_col;
            Bs[load_row][load_col] =
                (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // --- Compute: each thread does TM * BK FMAs ---
        #pragma unroll
        for (int k = 0; k < BK4; k++) {
            float b_val = Bs[k][tx];  // Reused across TM rows
            #pragma unroll
            for (int m = 0; m < TM4; m++) {
                accum[m] += As[ty * TM4 + m][k] * b_val;
            }
        }

        __syncthreads();
    }

    // Write results
    for (int m = 0; m < TM4; m++) {
        int g_row = row_base + m;
        if (g_row < M && col < N) {
            C[g_row * N + col] = accum[m];
        }
    }
}

void launch_gemm_1d_blocktile_stream(const float *A, const float *B, float *C,
                                     int M, int N, int K, cudaStream_t stream) {
    dim3 block(BN4, BM4 / TM4);  // (64, 8) = 512 threads
    dim3 grid((N + BN4 - 1) / BN4, (M + BM4 - 1) / BM4);
    gemm_1d_blocktile<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

void launch_gemm_1d_blocktile(const float *A, const float *B, float *C,
                              int M, int N, int K) {
    launch_gemm_1d_blocktile_stream(A, B, C, M, N, K, 0);
}
