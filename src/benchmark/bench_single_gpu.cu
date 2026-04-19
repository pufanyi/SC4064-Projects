// Copyright 2025 Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au
// SC4064 GPU Programming, Nanyang Technological University
//
// bench_single_gpu.cu - Single-GPU GEMM benchmark -- correctness + timing.
//
// Timing:  each of `repeat` iterations is measured with its own cudaEvent
//          pair, giving `repeat` independent samples for stats (mean,
//          median, stddev, min, max).

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../kernels/kernel_registry.cuh"
#include "../utils/bench_stats.cuh"
#include "../utils/cuda_raii.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/device_matrix.cuh"

// Measure one sample (ms) with a freshly-recorded CUDA event pair.
static inline double time_one_launch(const GemmKernel& kernel, const float* dA, const float* dB,
                                     float* dC, int M, int N, int K, cudaEvent_t start,
                                     cudaEvent_t stop) {
    CUDA_CHECK(cudaEventRecord(start, 0));
    kernel.launch(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}

// Returns per-iteration stats.  Each of `repeat` iters = one CUDA-event
// measurement of one kernel launch (NOT averaged internally).
static BenchStats benchmark_kernel(const GemmKernel& kernel, const float* dA, const float* dB,
                                   float* dC, int M, int N, int K, int warmup, int repeat) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; i++) kernel.launch(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> samples;
    samples.reserve(repeat);
    for (int i = 0; i < repeat; i++) {
        samples.push_back(time_one_launch(kernel, dA, dB, dC, M, N, K, start, stop));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return compute_stats(samples);
}

int main(int argc, char** argv) {
    // Line-buffered stdout so piping to a file / tee still streams output.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    // Optional: env-var knobs so smoke tests can run fast.
    const int kWarmup = std::atoi(std::getenv("BENCH_WARMUP") ? std::getenv("BENCH_WARMUP") : "5");
    const int kRepeat = std::atoi(std::getenv("BENCH_REPEAT") ? std::getenv("BENCH_REPEAT") : "20");

    print_device_info();

    CublasHandle handle;
    for (int i = 0; i < KernelRegistry::count(); i++) {
        GemmKernel& k = KernelRegistry::get_mut(i);
        if (k.needs_cublas()) k.set_cublas_handle(handle);
    }

    // --- Correctness verification ---
    printf("===== Correctness Verification (M=N=K=256) =====\n");
    {
        constexpr int M = 256, N = 256, K = 256;
        std::vector<float> hA(M * K), hB(K * N), hC_ref(M * N), hC_gpu(M * N);
        init_matrix(hA.data(), M, K, 42);
        init_matrix(hB.data(), K, N, 137);
        cpu_gemm(hA.data(), hB.data(), hC_ref.data(), M, N, K);

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

    // --- Performance ---
    // Collect stats into a (kernel, size) grid first -- we need the full grid
    // to print the stddev and detail tables afterwards -- but ALSO stream each
    // kernel's GFLOPS row as soon as it's computed so users see progress.
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096, 8192, 16384};
    std::vector<std::vector<BenchStats>> all_stats(KernelRegistry::count());

    // Pre-alloc + init per-size device matrices (reuse across all kernels)
    std::vector<DeviceMatrix> dA_per_size, dB_per_size, dC_per_size;
    dA_per_size.reserve(sizes.size());
    dB_per_size.reserve(sizes.size());
    dC_per_size.reserve(sizes.size());
    for (int s : sizes) {
        DeviceMatrix a(s, s), b(s, s), c(s, s);
        a.init_random(42);
        b.init_random(43);
        dA_per_size.push_back(std::move(a));
        dB_per_size.push_back(std::move(b));
        dC_per_size.push_back(std::move(c));
    }

    printf("\n===== Performance Benchmark (GFLOPS, mean of %d) =====\n", kRepeat);
    printf("%-20s", "Kernel");
    for (int s : sizes) printf("  %6d", s);
    printf("\n");
    for (int i = 0; i < 20 + 8 * (int)sizes.size(); i++) printf("-");
    printf("\n");

    for (int k = 0; k < KernelRegistry::count(); k++) {
        all_stats[k].resize(sizes.size());
        printf("%-20s", KernelRegistry::get(k).name());
        for (size_t si = 0; si < sizes.size(); si++) {
            int s = sizes[si];
            all_stats[k][si] = benchmark_kernel(KernelRegistry::get(k), dA_per_size[si].get(),
                                                dB_per_size[si].get(), dC_per_size[si].get(), s, s,
                                                s, kWarmup, kRepeat);
            double m = all_stats[k][si].mean;
            double gflops = (m > 0) ? gemm_gflops(s, s, s, static_cast<float>(m)) : 0.0;
            printf("  %6.0f", gflops);
        }
        printf("\n");
    }

    // Table 2: relative stddev (coefficient of variation, %)
    printf("\n===== Performance Stddev (CoV %%, n=%d) =====\n", kRepeat);
    printf("%-20s", "Kernel");
    for (int s : sizes) printf("  %6d", s);
    printf("\n");
    for (int i = 0; i < 20 + 8 * (int)sizes.size(); i++) printf("-");
    printf("\n");
    for (int k = 0; k < KernelRegistry::count(); k++) {
        printf("%-20s", KernelRegistry::get(k).name());
        for (size_t si = 0; si < sizes.size(); si++) {
            double m = all_stats[k][si].mean;
            double cov = (m > 0) ? 100.0 * all_stats[k][si].stddev / m : 0.0;
            printf("  %6.2f", cov);
        }
        printf("\n");
    }

    // Table 3: detail at M=N=K=4096.  Picked so all kernels (including the
    // slow naive/uncoalesced ones) have reached steady-state GFLOPS -- at
    // 2048 cuBLAS runs in ~0.35 ms (near timer noise floor) and the naive
    // kernels haven't climbed to their peak yet, so the stats aren't
    // comparable across rows.
    constexpr int kDetailSize = 4096;
    printf("\n===== Detailed Timing at M=N=K=%d (ms, n=%d) =====\n", kDetailSize, kRepeat);
    printf("%-20s  %10s  %10s  %10s  %10s  %10s\n", "Kernel", "mean", "median", "stddev", "min",
           "max");
    for (int i = 0; i < 80; i++) printf("-");
    printf("\n");
    int mid_idx = -1;
    for (size_t si = 0; si < sizes.size(); si++)
        if (sizes[si] == kDetailSize) mid_idx = static_cast<int>(si);
    if (mid_idx >= 0) {
        for (int k = 0; k < KernelRegistry::count(); k++) {
            const auto& s = all_stats[k][mid_idx];
            printf("%-20s  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n", KernelRegistry::get(k).name(),
                   s.mean, s.median, s.stddev, s.min_v, s.max_v);
        }
    }

    printf("\nDone.\n");
    return 0;
}
