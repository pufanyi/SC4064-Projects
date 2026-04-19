// Copyright 2025 Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au
// SC4064 GPU Programming, Nanyang Technological University
//
// bench_multi_gpu.cu - Multi-GPU tensor parallelism benchmark -- 6 scaling experiments.

#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thread>
#include <vector>

#include "../kernels/kernel_registry.cuh"
#include "../tensor_parallel/tensor_parallel.cuh"
#include "../utils/bench_stats.cuh"
#include "../utils/cuda_raii.cuh"
#include "../utils/cuda_utils.cuh"
#include "../utils/device_matrix.cuh"
#include "../utils/nccl_utils.cuh"

// ============================================================================
// RAII Device Context — owns per-GPU cuBLAS handle and CUDA streams
// ============================================================================
struct DeviceContext {
    int device_id;
    CublasHandle handle;
    CudaStream compute_stream;
    CudaStream comm_stream;

    // The caller MUST cudaSetDevice(dev) BEFORE constructing -- member
    // initialisers run before the constructor body, so handle and streams
    // would otherwise bind to whatever the current device happened to be.
    explicit DeviceContext(int dev) : device_id(dev) {}

    /// Activate this GPU context on the current thread: set device + cuBLAS handle.
    void activate() const {
        CUDA_CHECK(cudaSetDevice(device_id));
        for (int i = 0; i < KernelRegistry::count(); i++) {
            auto& k = KernelRegistry::get_mut(i);
            if (k.needs_cublas()) {
                k.set_cublas_handle(handle);
            }
        }
    }
};

// ============================================================================
// NCCL Communicator Registry — pre-initializes comm groups for all GPU counts
// ============================================================================
class CommRegistry {
    std::map<int, std::vector<ncclComm_t>> groups_;

   public:
    explicit CommRegistry(int max_gpus) {
        std::vector<int> devs(max_gpus);
        for (int i = 0; i < max_gpus; i++) devs[i] = i;

        for (int p = 2; p <= max_gpus; p *= 2) {
            groups_[p].resize(p);
            NCCL_CHECK(ncclCommInitAll(groups_[p].data(), p, devs.data()));
        }
        if (max_gpus > 1 && (max_gpus & (max_gpus - 1)) != 0) {
            groups_[max_gpus].resize(max_gpus);
            NCCL_CHECK(ncclCommInitAll(groups_[max_gpus].data(), max_gpus, devs.data()));
        }

        // Provide cuBLAS handles to kernels that need them (e.g. cuBLAS kernel).
        // We use GPU 0's handle; each per-GPU context sets its own device before launch.
        for (int i = 0; i < KernelRegistry::count(); i++) {
            auto& k = KernelRegistry::get_mut(i);
            if (k.needs_cublas()) {
                // Handle will be set per-GPU at call site; this is a placeholder
                // so the kernel object knows cublas is available.
            }
        }
    }

    ~CommRegistry() {
        for (auto& [count, comms] : groups_) {
            for (ncclComm_t comm : comms) {
                if (comm) ncclCommDestroy(comm);
            }
        }
    }

    CommRegistry(const CommRegistry&) = delete;
    CommRegistry& operator=(const CommRegistry&) = delete;

    ncclComm_t get(int active_gpus, int gpu_id) const {
        if (active_gpus <= 1) return nullptr;
        return groups_.at(active_gpus)[gpu_id];
    }
};

// ============================================================================
// Multi-GPU execution helpers
// ============================================================================
template <typename Fn>
void run_on_gpus(int active_gpus, Fn fn) {
    std::vector<std::thread> workers;
    workers.reserve(active_gpus);
    for (int g = 0; g < active_gpus; g++) {
        workers.emplace_back([&, g]() { fn(g); });
    }
    for (auto& worker : workers) worker.join();
}

template <typename Fn>
BenchStats benchmark_stats(int active_gpus, int warmup, int repeat, Fn fn) {
    for (int i = 0; i < warmup; i++) run_on_gpus(active_gpus, fn);

    std::vector<double> samples;
    samples.reserve(repeat);
    for (int i = 0; i < repeat; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        run_on_gpus(active_gpus, fn);
        auto stop = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    return compute_stats(samples);
}

std::vector<int> build_scaling_counts(int max_gpus) {
    std::vector<int> counts = {1};
    for (int p = 2; p <= max_gpus; p *= 2) counts.push_back(p);
    if (counts.back() != max_gpus) counts.push_back(max_gpus);
    return counts;
}

// ============================================================================
// RAII Buffer structs — automatic allocation, initialization, and cleanup
// ============================================================================

// Column-parallel buffers: used by Experiments 1, 2, 3, 4
struct ColParallelBuffers {
    int device_id;
    DeviceMatrix X, W, Y, Y_full;

    ColParallelBuffers(int gpu_id, int M, int K, int N_local, int N) : device_id(gpu_id) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        X = DeviceMatrix(M, K);
        W = DeviceMatrix(K, N_local);
        Y = DeviceMatrix(M, N_local);
        Y_full = DeviceMatrix(M, N);
        X.init_random(42);
        W.init_random(137 + gpu_id);
    }

    ~ColParallelBuffers() { CUDA_CHECK(cudaSetDevice(device_id)); }

    ColParallelBuffers(ColParallelBuffers&&) = default;
    ColParallelBuffers& operator=(ColParallelBuffers&&) = default;
};

// MLP buffers: used by Experiment 5 (forward + backward)
struct MLPBuffers {
    int device_id;
    DeviceMatrix X, W1, W2, Hidden, YPartial, Y;
    DeviceMatrix dY, dW1, dW2, dHidden, dXPartial, dX;

    MLPBuffers(int gpu_id, int M, int K, int H_local, int N) : device_id(gpu_id) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        X = DeviceMatrix(M, K);
        W1 = DeviceMatrix(K, H_local);
        W2 = DeviceMatrix(H_local, N);
        Hidden = DeviceMatrix(M, H_local);
        YPartial = DeviceMatrix(M, N);
        Y = DeviceMatrix(M, N);
        dY = DeviceMatrix(M, N);
        dW1 = DeviceMatrix(K, H_local);
        dW2 = DeviceMatrix(H_local, N);
        dHidden = DeviceMatrix(M, H_local);
        dXPartial = DeviceMatrix(M, K);
        dX = DeviceMatrix(M, K);
        X.init_random(42);
        W1.init_random(137 + gpu_id);
        W2.init_random(271 + gpu_id);
        dY.init_random(314 + gpu_id);
    }

    ~MLPBuffers() { CUDA_CHECK(cudaSetDevice(device_id)); }

    MLPBuffers(MLPBuffers&&) = default;
    MLPBuffers& operator=(MLPBuffers&&) = default;
};

// Row-parallel buffers: used by Experiment 6 (overlap comparison)
struct RowParallelBuffers {
    int device_id;
    DeviceMatrix X, W, Y, YReduced;
    DeviceMatrix YOverlap, YReducedOverlap;

    RowParallelBuffers(int gpu_id, int M, int K_local, int N) : device_id(gpu_id) {
        CUDA_CHECK(cudaSetDevice(gpu_id));
        X = DeviceMatrix(M, K_local);
        W = DeviceMatrix(K_local, N);
        Y = DeviceMatrix(M, N);
        YReduced = DeviceMatrix(M, N);
        YOverlap = DeviceMatrix(M, N);
        YReducedOverlap = DeviceMatrix(M, N);
        X.init_random(42);
        W.init_random(137 + gpu_id);
    }

    ~RowParallelBuffers() { CUDA_CHECK(cudaSetDevice(device_id)); }

    RowParallelBuffers(RowParallelBuffers&&) = default;
    RowParallelBuffers& operator=(RowParallelBuffers&&) = default;
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int max_gpus = 2;
    int kernel_id = KernelRegistry::count() - 1;  // default: last registered (cuBLAS)

    if (argc > 1) max_gpus = atoi(argv[1]);
    if (argc > 2) {
        int kid = atoi(argv[2]);
        if (kid >= 0 && kid < KernelRegistry::count()) {
            kernel_id = kid;
        }
    }

    const GemmKernel& kernel = KernelRegistry::get(kernel_id);

    int available_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    if (available_gpus <= 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }
    max_gpus = std::min(max_gpus, available_gpus);

    printf("===== Multi-GPU Tensor Parallel Benchmark =====\n");
    printf("Max GPUs used:      %d\n", max_gpus);
    printf("Local GEMM kernel:  %s\n\n", kernel.name());

    // Initialize per-GPU contexts (RAII: handle + streams auto-managed).
    // cudaSetDevice BEFORE emplace_back so the handle + streams are bound
    // to the right device at construction time.
    std::vector<DeviceContext> contexts;
    contexts.reserve(max_gpus);
    for (int g = 0; g < max_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        contexts.emplace_back(g);
        print_device_info();
    }

    // Activate GPU 0 as default for main thread
    contexts[0].activate();

    CommRegistry comms(max_gpus);
    const auto scaling_counts = build_scaling_counts(max_gpus);
    // col_sizes includes 49152 (37 GB/GPU at 1 GPU); 65536 would OOM on 1 GPU.
    // mlp_sizes / ovlp_sizes stay at 32768 because of the 6/4 full M*N
    // buffers per GPU (memory bound, not compute bound).
    const std::vector<int> col_sizes  = {2048, 4096, 8192, 16384, 32768, 49152};
    const std::vector<int> mlp_sizes  = {2048, 4096, 8192, 16384, 32768};
    const std::vector<int> ovlp_sizes = {2048, 4096, 8192, 16384, 32768};
    const std::vector<int> weak_sizes = {2048, 4096, 8192};
    constexpr int kWarmup = 5;
    constexpr int kRepeat = 20;

    // =====================================================================
    // Experiment 1: Strong Scaling
    // =====================================================================
    printf("===== Exp 1: Strong Scaling — Column Parallel Forward =====\n");
    printf("%-6s %-6s %-6s %-6s  %10s %10s  %10s %10s  %10s %10s  %8s\n", "M", "N", "K", "GPUs",
           "GEMM(ms)", "GEMM_std", "Comm(ms)", "Comm_std", "Total(ms)", "Total_std", "GFLOPS");
    printf("--------------------------------------------------------------------------"
           "-----------------------\n");

    for (int active_gpus : scaling_counts) {
        for (int S : col_sizes) {
            const int M = S, N = S, K = S;
            const int N_local = N / active_gpus;

            std::vector<ColParallelBuffers> bufs;
            bufs.reserve(active_gpus);
            for (int g = 0; g < active_gpus; g++) bufs.emplace_back(g, M, K, N_local, N);

            BenchStats gemm = benchmark_stats(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();

                kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(), M, N_local, K,
                              ctx.compute_stream);
                ctx.compute_stream.synchronize();
            });

            BenchStats comm{};
            if (active_gpus > 1) {
                comm = benchmark_stats(active_gpus, kWarmup, kRepeat, [&](int g) {
                    auto& ctx = contexts[g];
                    ctx.activate();

                    NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local,
                                             ncclFloat, comms.get(active_gpus, g),
                                             ctx.comm_stream));
                    ctx.comm_stream.synchronize();
                });
            }

            BenchStats total = benchmark_stats(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();

                column_parallel_forward(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                                        bufs[g].Y_full.get(), M, N, K, active_gpus, g, ctx.handle,
                                        comms.get(active_gpus, g), ctx.compute_stream, kernel);
                ctx.compute_stream.synchronize();
            });

            printf("%-6d %-6d %-6d %-6d  %10.3f %10.3f  %10.3f %10.3f  %10.3f %10.3f  %8.1f\n", M,
                   N, K, active_gpus, gemm.mean, gemm.stddev, comm.mean, comm.stddev, total.mean,
                   total.stddev, gemm_gflops(M, N, K, total.mean));
        }
    }

    // =====================================================================
    // Experiment 2: Weak Scaling
    // =====================================================================
    printf("\n===== Exp 2: Weak Scaling — per-GPU tile sweep × GPU count =====\n");
    printf("%-6s %-6s %-6s %-6s  %10s %10s  %10s %10s  %10s %10s  %8s\n", "M", "N_tot", "K",
           "GPUs", "GEMM(ms)", "GEMM_std", "Comm(ms)", "Comm_std", "Total(ms)", "Total_std",
           "GFLOPS");
    printf("--------------------------------------------------------------------------"
           "-----------------------\n");

    for (int active_gpus : scaling_counts) {
        for (int per_gpu : weak_sizes) {
            const int M = per_gpu, K = per_gpu, N_local = per_gpu;
            const int N_total = N_local * active_gpus;

            std::vector<ColParallelBuffers> bufs;
            bufs.reserve(active_gpus);
            for (int g = 0; g < active_gpus; g++) bufs.emplace_back(g, M, K, N_local, N_total);

            BenchStats gemm = benchmark_stats(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();
                kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(), M, N_local, K,
                              ctx.compute_stream);
                ctx.compute_stream.synchronize();
            });

            BenchStats comm{};
            if (active_gpus > 1) {
                comm = benchmark_stats(active_gpus, kWarmup, kRepeat, [&](int g) {
                    auto& ctx = contexts[g];
                    ctx.activate();

                    NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local,
                                             ncclFloat, comms.get(active_gpus, g),
                                             ctx.comm_stream));
                    ctx.comm_stream.synchronize();
                });
            }

            BenchStats total = benchmark_stats(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();
                column_parallel_forward(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                                        bufs[g].Y_full.get(), M, N_total, K, active_gpus, g,
                                        ctx.handle, comms.get(active_gpus, g), ctx.compute_stream,
                                        kernel);
                ctx.compute_stream.synchronize();
            });

            printf("%-6d %-6d %-6d %-6d  %10.3f %10.3f  %10.3f %10.3f  %10.3f %10.3f  %8.1f\n", M,
                   N_total, K, active_gpus, gemm.mean, gemm.stddev, comm.mean, comm.stddev,
                   total.mean, total.stddev, gemm_gflops(M, N_total, K, total.mean));
        }
    }

    // =====================================================================
    // Experiment 3: Comm/Compute Ratio vs Matrix Size
    // =====================================================================
    printf("\n===== Exp 3: Comm/Compute Ratio vs Matrix Size (%d GPUs) =====\n", max_gpus);
    printf("%-6s  %10s %10s  %10s %10s  %8s\n", "Size", "GEMM(ms)", "GEMM_std", "Comm(ms)",
           "Comm_std", "Ratio");
    printf("----------------------------------------------------------------------\n");

    for (int S : col_sizes) {
        const int M = S, N = S, K = S;
        const int N_local = N / max_gpus;

        std::vector<ColParallelBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++) bufs.emplace_back(g, M, K, N_local, N);

        BenchStats gemm = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(), M, N_local, K,
                          ctx.compute_stream);
            ctx.compute_stream.synchronize();
        });

        BenchStats comm = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local, ncclFloat,
                                     comms.get(max_gpus, g), ctx.comm_stream));
            ctx.comm_stream.synchronize();
        });

        printf("%-6d  %10.3f %10.3f  %10.3f %10.3f  %8.2f\n", S, gemm.mean, gemm.stddev, comm.mean,
               comm.stddev, gemm.mean > 0 ? comm.mean / gemm.mean : 0.0);
    }

    // =====================================================================
    // Experiment 4: Comm/Compute Ratio across Different Kernels
    // =====================================================================
    printf("\n===== Exp 4: Comm/Compute Ratio across Kernels (size=4096, %d GPUs) =====\n",
           max_gpus);
    printf("%-20s  %10s %10s  %10s %10s  %8s\n", "Kernel", "GEMM(ms)", "GEMM_std", "Comm(ms)",
           "Comm_std", "Ratio");
    printf("----------------------------------------------------------------------------\n");

    {
        constexpr int S = 4096;
        const int M = S, N = S, K = S;
        const int N_local = N / max_gpus;

        std::vector<ColParallelBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++) bufs.emplace_back(g, M, K, N_local, N);

        BenchStats comm = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local, ncclFloat,
                                     comms.get(max_gpus, g), ctx.comm_stream));
            ctx.comm_stream.synchronize();
        });

        for (const auto& kptr : KernelRegistry::all()) {
            const GemmKernel& test_kernel = *kptr;

            BenchStats gemm = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();

                test_kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(), M, N_local, K,
                                   ctx.compute_stream);
                ctx.compute_stream.synchronize();
            });

            printf("%-20s  %10.3f %10.3f  %10.3f %10.3f  %8.2f\n", test_kernel.name(), gemm.mean,
                   gemm.stddev, comm.mean, comm.stddev,
                   gemm.mean > 0 ? comm.mean / gemm.mean : 0.0);
        }
    }

    // =====================================================================
    // Experiment 5: Parallel MLP Forward + Backward
    // =====================================================================
    printf("\n===== Exp 5: Parallel MLP Forward + Backward (%d GPUs) =====\n", max_gpus);
    printf("%-6s %-6s %-6s %-6s  %10s %10s  %10s %10s  %10s\n", "M", "H", "N", "GPUs", "Fwd(ms)",
           "Fwd_std", "Bwd(ms)", "Bwd_std", "Total(ms)");
    printf("------------------------------------------------------------------------"
           "-----\n");

    for (int S : sizes) {
        const int M = S, K = S, H = S, N = S;
        const int H_local = H / max_gpus;

        std::vector<MLPBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++) bufs.emplace_back(g, M, K, H_local, N);

        BenchStats fwd = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            parallel_mlp_forward(bufs[g].X.get(), bufs[g].W1.get(), bufs[g].W2.get(),
                                 bufs[g].Hidden.get(), bufs[g].YPartial.get(), bufs[g].Y.get(), M,
                                 K, H, N, max_gpus, g, ctx.handle, comms.get(max_gpus, g),
                                 ctx.compute_stream, kernel);
            ctx.compute_stream.synchronize();
        });

        BenchStats bwd = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            parallel_mlp_backward(bufs[g].X.get(), bufs[g].W1.get(), bufs[g].W2.get(),
                                  bufs[g].Hidden.get(), bufs[g].dY.get(), bufs[g].dW1.get(),
                                  bufs[g].dW2.get(), bufs[g].dHidden.get(), bufs[g].dXPartial.get(),
                                  bufs[g].dX.get(), M, K, H, N, max_gpus, g, ctx.handle,
                                  comms.get(max_gpus, g), ctx.compute_stream, kernel);
            ctx.compute_stream.synchronize();
        });

        printf("%-6d %-6d %-6d %-6d  %10.3f %10.3f  %10.3f %10.3f  %10.3f\n", M, H, N, max_gpus,
               fwd.mean, fwd.stddev, bwd.mean, bwd.stddev, fwd.mean + bwd.mean);
    }

    // =====================================================================
    // Experiment 6: Communication-Compute Overlap
    // =====================================================================
    printf("\n===== Exp 6: Row Parallel — No Overlap vs Overlap (%d GPUs) =====\n", max_gpus);
    printf("%-6s  %-8s  %10s %10s  %10s %10s  %8s\n", "Size", "Chunks", "NoOvlp(ms)", "NoOvlp_std",
           "Overlap(ms)", "Overlap_std", "Speedup");
    printf("------------------------------------------------------------------------"
           "---\n");

    for (int S : sizes) {
        const int M = S, N = S, K = S;
        const int K_local = K / max_gpus;
        constexpr int num_chunks = 4;

        std::vector<RowParallelBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++) bufs.emplace_back(g, M, K_local, N);

        BenchStats no_overlap = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            row_parallel_forward(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                                 bufs[g].YReduced.get(), M, N, K, max_gpus, g, ctx.handle,
                                 comms.get(max_gpus, g), ctx.compute_stream, kernel);
            ctx.compute_stream.synchronize();
        });

        BenchStats overlap = benchmark_stats(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            row_parallel_forward_overlap(bufs[g].X.get(), bufs[g].W.get(), bufs[g].YOverlap.get(),
                                         bufs[g].YReducedOverlap.get(), M, N, K, max_gpus, g,
                                         num_chunks, ctx.handle, comms.get(max_gpus, g),
                                         ctx.compute_stream, ctx.comm_stream, kernel);
            ctx.compute_stream.synchronize();
            ctx.comm_stream.synchronize();
        });

        printf("%-6d  %-8d  %10.3f %10.3f  %10.3f %10.3f  %8.2fx\n", S, num_chunks, no_overlap.mean,
               no_overlap.stddev, overlap.mean, overlap.stddev,
               overlap.mean > 0 ? no_overlap.mean / overlap.mean : 0.0);
    }

    printf("\nDone.\n");
    return 0;
}
