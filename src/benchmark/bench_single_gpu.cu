/*
 * Single-GPU GEMM Benchmark
 * ==========================
 * Tests each kernel for correctness against CPU reference (small size)
 * and benchmarks GFLOPS across multiple matrix sizes.
 *
 * Usage: ./bench_single_gpu [--verify-only] [--size M,N,K]
 */

#include <cublas_v2.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../kernels/kernels.cuh"
#include "../utils/cuda_utils.cuh"

// Also expose uncoalesced for comparison
extern void launch_gemm_uncoalesced(const float* A, const float* B, float* C, int M, int N, int K);

struct KernelInfo {
    const char* name;
    void (*launch)(const float*, const float*, float*, int, int, int);
};

// Benchmark a kernel: warm up, then average over multiple runs
double benchmark_kernel(void (*launch)(const float*, const float*, float*, int, int, int),
                        const float* dA, const float* dB, float* dC, int M, int N, int K,
                        int warmup = 3, int repeat = 10) {
    GpuTimer timer;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        launch(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    timer.tic();
    for (int i = 0; i < repeat; i++) {
        launch(dA, dB, dC, M, N, K);
    }
    float total_ms = timer.toc();
    return gemm_gflops(M, N, K, total_ms / repeat);
}

double benchmark_cublas(cublasHandle_t handle, const float* dA, const float* dB, float* dC, int M,
                        int N, int K, int warmup = 3, int repeat = 10) {
    GpuTimer timer;
    for (int i = 0; i < warmup; i++) {
        launch_cublas_gemm(handle, dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.tic();
    for (int i = 0; i < repeat; i++) {
        launch_cublas_gemm(handle, dA, dB, dC, M, N, K);
    }
    float total_ms = timer.toc();
    return gemm_gflops(M, N, K, total_ms / repeat);
}

int main(int argc, char** argv) {
    print_device_info();

    // --- Correctness verification (small matrix) ---
    printf("===== Correctness Verification (M=N=K=256) =====\n");
    {
        const int M = 256, N = 256, K = 256;
        size_t sizeA = M * K * sizeof(float);
        size_t sizeB = K * N * sizeof(float);
        size_t sizeC = M * N * sizeof(float);

        float* hA = (float*)malloc(sizeA);
        float* hB = (float*)malloc(sizeB);
        float* hC_ref = (float*)malloc(sizeC);
        float* hC_gpu = (float*)malloc(sizeC);

        init_matrix(hA, M, K, 42);
        init_matrix(hB, K, N, 137);
        cpu_gemm(hA, hB, hC_ref, M, N, K);

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, sizeA));
        CUDA_CHECK(cudaMalloc(&dB, sizeB));
        CUDA_CHECK(cudaMalloc(&dC, sizeC));
        CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

        KernelInfo kernels[] = {
            {"1_naive", launch_gemm_naive},
            {"2_coalesced", launch_gemm_coalesced},
            {"2_uncoalesced", launch_gemm_uncoalesced},
            {"3_smem", launch_gemm_smem},
            {"4_1d_blocktile", launch_gemm_1d_blocktile},
            {"5_2d_blocktile", launch_gemm_2d_blocktile},
            {"6_vectorized", launch_gemm_vectorized},
            {"7_warptile", launch_gemm_warptile},
        };

        for (auto& ki : kernels) {
            CUDA_CHECK(cudaMemset(dC, 0, sizeC));
            ki.launch(dA, dB, dC, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(hC_gpu, dC, sizeC, cudaMemcpyDeviceToHost));
            printf("%-20s: ", ki.name);
            verify(hC_ref, hC_gpu, M * N);
        }

        // cuBLAS
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUDA_CHECK(cudaMemset(dC, 0, sizeC));
        launch_cublas_gemm(handle, dA, dB, dC, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hC_gpu, dC, sizeC, cudaMemcpyDeviceToHost));
        printf("%-20s: ", "cuBLAS");
        verify(hC_ref, hC_gpu, M * N);
        cublasDestroy(handle);

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        free(hA);
        free(hB);
        free(hC_ref);
        free(hC_gpu);
    }

    // --- Performance benchmark ---
    printf("\n===== Performance Benchmark (GFLOPS) =====\n");
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192, 16384};

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Print header
    printf("%-20s", "Kernel");
    for (int s : sizes) printf("  %5d", s);
    printf("\n");
    for (int i = 0; i < 20 + 7 * (int)sizes.size(); i++) printf("-");
    printf("\n");

    KernelInfo kernels[] = {
        {"1_naive", launch_gemm_naive},
        {"2_coalesced", launch_gemm_coalesced},
        {"2_uncoalesced", launch_gemm_uncoalesced},
        {"3_smem", launch_gemm_smem},
        {"4_1d_blocktile", launch_gemm_1d_blocktile},
        {"5_2d_blocktile", launch_gemm_2d_blocktile},
        {"6_vectorized", launch_gemm_vectorized},
        {"7_warptile", launch_gemm_warptile},
    };

    // Also benchmark cuBLAS
    for (auto& ki : kernels) {
        printf("%-20s", ki.name);
        for (int s : sizes) {
            float *dA, *dB, *dC;
            size_t sA = s * s * sizeof(float);
            CUDA_CHECK(cudaMalloc(&dA, sA));
            CUDA_CHECK(cudaMalloc(&dB, sA));
            CUDA_CHECK(cudaMalloc(&dC, sA));

            // Init on device (just random, correctness already verified)
            float* h = (float*)malloc(sA);
            init_matrix(h, s, s, 42);
            CUDA_CHECK(cudaMemcpy(dA, h, sA, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, h, sA, cudaMemcpyHostToDevice));
            free(h);

            double gflops = benchmark_kernel(ki.launch, dA, dB, dC, s, s, s);
            printf("  %5.0f", gflops);

            CUDA_CHECK(cudaFree(dA));
            CUDA_CHECK(cudaFree(dB));
            CUDA_CHECK(cudaFree(dC));
        }
        printf("\n");
    }

    // cuBLAS row
    printf("%-20s", "cuBLAS");
    for (int s : sizes) {
        float *dA, *dB, *dC;
        size_t sA = s * s * sizeof(float);
        CUDA_CHECK(cudaMalloc(&dA, sA));
        CUDA_CHECK(cudaMalloc(&dB, sA));
        CUDA_CHECK(cudaMalloc(&dC, sA));
        float* h = (float*)malloc(sA);
        init_matrix(h, s, s, 42);
        CUDA_CHECK(cudaMemcpy(dA, h, sA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, h, sA, cudaMemcpyHostToDevice));
        free(h);
        double gflops = benchmark_cublas(handle, dA, dB, dC, s, s, s);
        printf("  %5.0f", gflops);
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
    printf("\n");

    cublasDestroy(handle);
    printf("\nDone.\n");
    return 0;
}
