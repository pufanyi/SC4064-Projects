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

#include "../kernels/kernel_registry.cuh"
#include "../utils/cuda_raii.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/device_matrix.cuh"

// Benchmark a kernel: warm up, then average over multiple runs
double benchmark_kernel(const GemmKernel& kernel,
                        const float* dA, const float* dB, float* dC, int M, int N, int K,
                        int warmup = 3, int repeat = 10) {
    GpuTimer timer;

    for (int i = 0; i < warmup; i++) {
        kernel.launch(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer.tic();
    for (int i = 0; i < repeat; i++) {
        kernel.launch(dA, dB, dC, M, N, K);
    }
    float total_ms = timer.toc();
    return gemm_gflops(M, N, K, total_ms / repeat);
}

int main(int argc, char** argv) {
    print_device_info();

    // Create a cuBLAS handle and inject it into any kernel that needs one
    CublasHandle handle;
    for (int i = 0; i < KernelRegistry::count(); i++) {
        GemmKernel& k = KernelRegistry::get_mut(i);
        if (k.needs_cublas()) {
            k.set_cublas_handle(handle);
        }
    }

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

        for (auto& kptr : KernelRegistry::all()) {
            dC.zero();
            kptr->launch(dA.get(), dB.get(), dC.get(), M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());
            dC.copy_to_host(hC_gpu.data());
            printf("%-20s: ", kptr->name());
            verify(hC_ref.data(), hC_gpu.data(), M * N);
        }
    }

    // --- Performance benchmark ---
    printf("\n===== Performance Benchmark (GFLOPS) =====\n");
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192, 16384};

    // Print header
    printf("%-20s", "Kernel");
    for (int s : sizes) printf("  %5d", s);
    printf("\n");
    for (int i = 0; i < 20 + 7 * (int)sizes.size(); i++) printf("-");
    printf("\n");

    for (auto& kptr : KernelRegistry::all()) {
        printf("%-20s", kptr->name());
        for (int s : sizes) {
            DeviceMatrix dA(s, s), dB(s, s), dC(s, s);
            dA.init_random(42);
            dB.init_random(42);

            double gflops = benchmark_kernel(*kptr, dA.get(), dB.get(), dC.get(), s, s, s);
            printf("  %5.0f", gflops);
        }
        printf("\n");
    }

    printf("\nDone.\n");
    return 0;
}
