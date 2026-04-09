/*
 * Single-GPU GEMM Benchmark
 * ==========================
 * Tests each kernel for correctness against CPU reference (small size)
 * and benchmarks GFLOPS across multiple matrix sizes.
 *
 * Usage: ./bench_single_gpu [--verify-only] [--size M,N,K]
 */

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../kernels/kernels.cuh"
#include "../utils/cuda_raii.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/device_matrix.cuh"

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

    for (int i = 0; i < warmup; i++) {
        launch(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

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
        constexpr int M = 256, N = 256, K = 256;

        // Host matrices
        std::vector<float> hA(M * K), hB(K * N), hC_ref(M * N), hC_gpu(M * N);
        init_matrix(hA.data(), M, K, 42);
        init_matrix(hB.data(), K, N, 137);
        cpu_gemm(hA.data(), hB.data(), hC_ref.data(), M, N, K);

        // Device matrices
        DeviceMatrix dA(M, K), dB(K, N), dC(M, N);
        dA.copy_from_host(hA.data());
        dB.copy_from_host(hB.data());

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
            dC.zero();
            ki.launch(dA.get(), dB.get(), dC.get(), M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            dC.copy_to_host(hC_gpu.data());
            printf("%-20s: ", ki.name);
            verify(hC_ref.data(), hC_gpu.data(), M * N);
        }

        // cuBLAS
        CublasHandle handle;
        dC.zero();
        launch_cublas_gemm(handle, dA.get(), dB.get(), dC.get(), M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        dC.copy_to_host(hC_gpu.data());
        printf("%-20s: ", "cuBLAS");
        verify(hC_ref.data(), hC_gpu.data(), M * N);
    }

    // --- Performance benchmark ---
    printf("\n===== Performance Benchmark (GFLOPS) =====\n");
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192, 16384};

    CublasHandle handle;

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

    for (auto& ki : kernels) {
        printf("%-20s", ki.name);
        for (int s : sizes) {
            DeviceMatrix dA(s, s), dB(s, s), dC(s, s);
            dA.init_random(42);
            dB.init_random(42);

            double gflops = benchmark_kernel(ki.launch, dA.get(), dB.get(), dC.get(), s, s, s);
            printf("  %5.0f", gflops);
        }
        printf("\n");
    }

    // cuBLAS row
    printf("%-20s", "cuBLAS");
    for (int s : sizes) {
        DeviceMatrix dA(s, s), dB(s, s), dC(s, s);
        dA.init_random(42);
        dB.init_random(42);

        double gflops = benchmark_cublas(handle, dA.get(), dB.get(), dC.get(), s, s, s);
        printf("  %5.0f", gflops);
    }
    printf("\n");

    printf("\nDone.\n");
    return 0;
}
