#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        cublasStatus_t stat = (call);                                                      \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                               \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)stat); \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

// ---------------------------------------------------------------------------
// Matrix utilities
// ---------------------------------------------------------------------------
inline void init_matrix(float* mat, int rows, int cols, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = dist(rng);
    }
}

inline void zero_matrix(float* mat, int n) {
    for (int i = 0; i < n; i++) mat[i] = 0.0f;
}

// CPU reference GEMM: C = A * B  (row-major, A: MxK, B: KxN, C: MxN)
inline void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify GPU result against reference with tolerance
inline bool verify(const float* ref, const float* gpu, int n, float atol = 1e-3f,
                   float rtol = 1e-3f) {
    int errors = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - gpu[i]);
        float tol = atol + rtol * fabsf(ref[i]);
        if (diff > tol) {
            if (errors < 5) {
                printf("  Mismatch at %d: ref=%.6f gpu=%.6f diff=%.6f\n", i, ref[i], gpu[i], diff);
            }
            errors++;
        }
        if (diff > max_diff) max_diff = diff;
    }
    if (errors > 0) {
        printf("  FAIL: %d / %d elements differ (max_diff=%.6f)\n", errors, n, max_diff);
        return false;
    }
    printf("  PASS (max_diff=%.6f)\n", max_diff);
    return true;
}

// ---------------------------------------------------------------------------
// Timing helper using CUDA events (RAII)
// ---------------------------------------------------------------------------
class GpuTimer {
    cudaEvent_t start_, stop_;

   public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void tic(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }

    float toc(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// Compute effective GFLOPS for GEMM: 2*M*N*K FLOPs
inline double gemm_gflops(int M, int N, int K, float ms) {
    double flops = 2.0 * (double)M * (double)N * (double)K;
    return flops / (ms * 1e6);  // ms -> seconds, flops -> GFLOPS
}

// Print device info
inline void print_device_info() {
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("=== Device %d: %s ===\n", dev, prop.name);
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    printf("  SM count:          %d\n", prop.multiProcessorCount);
    printf("  Clock rate:        %d MHz\n", clock_khz / 1000);
    printf("  Memory:            %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("  L2 Cache:          %d KB\n", prop.l2CacheSize / 1024);
    printf("  Shared mem/block:  %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Warp size:         %d\n", prop.warpSize);
    printf("  Compute cap:       %d.%d\n", prop.major, prop.minor);

    // Theoretical peak TFLOPS (FP32): SM_count * cores_per_SM * 2 * clock
    // A100: 108 SMs * 64 FP32 cores * 2 * 1.41 GHz ~ 19.5 TFLOPS
    double peak = (double)prop.multiProcessorCount * 64 * 2 * (clock_khz / 1e6);
    printf("  ~Peak FP32:        %.1f TFLOPS\n", peak / 1000.0);
    printf("\n");
}
