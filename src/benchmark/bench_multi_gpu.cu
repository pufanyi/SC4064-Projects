/*
 * Multi-GPU Tensor Parallelism Benchmark
 * ======================================
 * Runs true multi-rank NCCL experiments from a single process by launching
 * one host thread per active GPU. This keeps the benchmark self-contained
 * while ensuring every rank participates in collectives.
 *
 * Usage: ./bench_multi_gpu [max_num_gpus] [kernel_id]
 *   kernel_id: index into KernelRegistry (default: last registered, typically cuBLAS)
 */

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

    explicit DeviceContext(int dev) : device_id(dev) {
        CUDA_CHECK(cudaSetDevice(dev));
    }

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
double benchmark_wall_ms(int active_gpus, int warmup, int repeat, Fn fn) {
    for (int i = 0; i < warmup; i++) run_on_gpus(active_gpus, fn);

    double total_ms = 0.0;
    for (int i = 0; i < repeat; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        run_on_gpus(active_gpus, fn);
        auto stop = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(stop - start).count();
    }
    return total_ms / repeat;
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

    ColParallelBuffers(int gpu_id, int M, int K, int N_local, int N)
        : device_id(gpu_id) {
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

    MLPBuffers(int gpu_id, int M, int K, int H_local, int N)
        : device_id(gpu_id) {
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

    RowParallelBuffers(int gpu_id, int M, int K_local, int N)
        : device_id(gpu_id) {
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

    // Initialize per-GPU contexts (RAII: handle + streams auto-managed)
    std::vector<DeviceContext> contexts;
    contexts.reserve(max_gpus);
    for (int g = 0; g < max_gpus; g++) {
        contexts.emplace_back(g);
        print_device_info();
    }

    // Activate GPU 0 as default for main thread
    contexts[0].activate();

    CommRegistry comms(max_gpus);
    const auto scaling_counts = build_scaling_counts(max_gpus);
    const std::vector<int> sizes = {2048, 4096, 8192, 16384, 32768};
    constexpr int kWarmup = 2;
    constexpr int kRepeat = 5;

    // =====================================================================
    // Experiment 1: Strong Scaling
    // =====================================================================
    printf("===== Exp 1: Strong Scaling — Column Parallel Forward =====\n");
    printf("%-6s %-6s %-6s %-6s  %10s  %10s  %10s  %8s\n", "M", "N", "K", "GPUs", "GEMM(ms)",
           "Comm(ms)", "Total(ms)", "GFLOPS");
    printf("-----------------------------------------------------------------------\n");

    for (int active_gpus : scaling_counts) {
        for (int S : sizes) {
            const int M = S, N = S, K = S;
            const int N_local = N / active_gpus;

            std::vector<ColParallelBuffers> bufs;
            bufs.reserve(active_gpus);
            for (int g = 0; g < active_gpus; g++)
                bufs.emplace_back(g, M, K, N_local, N);

            double gemm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();

                kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                              M, N_local, K, ctx.compute_stream);
                ctx.compute_stream.synchronize();
            });

            double comm_ms = 0.0;
            if (active_gpus > 1) {
                comm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                    auto& ctx = contexts[g];
                    ctx.activate();

                    NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local,
                                             ncclFloat, comms.get(active_gpus, g),
                                             ctx.comm_stream));
                    ctx.comm_stream.synchronize();
                });
            }

            double total_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();

                column_parallel_forward(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                                        bufs[g].Y_full.get(), M, N, K, active_gpus, g, ctx.handle,
                                        comms.get(active_gpus, g), ctx.compute_stream, kernel);
                ctx.compute_stream.synchronize();
            });

            printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %10.3f  %8.1f\n", M, N, K, active_gpus,
                   gemm_ms, comm_ms, total_ms, gemm_gflops(M, N, K, total_ms));
        }
    }

    // =====================================================================
    // Experiment 2: Weak Scaling
    // =====================================================================
    printf("\n===== Exp 2: Weak Scaling — Fixed M=N_local=K=2048 per GPU =====\n");
    printf("%-6s %-6s %-6s %-6s  %10s  %10s  %10s  %8s\n", "M", "N_tot", "K", "GPUs", "GEMM(ms)",
           "Comm(ms)", "Total(ms)", "GFLOPS");
    printf("-----------------------------------------------------------------------\n");

    for (int active_gpus : scaling_counts) {
        constexpr int M = 2048, K = 2048, N_local = 2048;
        const int N_total = N_local * active_gpus;

        std::vector<ColParallelBuffers> bufs;
        bufs.reserve(active_gpus);
        for (int g = 0; g < active_gpus; g++)
            bufs.emplace_back(g, M, K, N_local, N_total);

        double gemm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                          M, N_local, K, ctx.compute_stream);
            ctx.compute_stream.synchronize();
        });

        double comm_ms = 0.0;
        if (active_gpus > 1) {
            comm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();

                NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local,
                                         ncclFloat, comms.get(active_gpus, g), ctx.comm_stream));
                ctx.comm_stream.synchronize();
            });
        }

        double total_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            column_parallel_forward(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                                    bufs[g].Y_full.get(), M, N_total, K, active_gpus, g,
                                    ctx.handle, comms.get(active_gpus, g), ctx.compute_stream,
                                    kernel);
            ctx.compute_stream.synchronize();
        });

        printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %10.3f  %8.1f\n", M, N_total, K,
               active_gpus, gemm_ms, comm_ms, total_ms, gemm_gflops(M, N_total, K, total_ms));
    }

    // =====================================================================
    // Experiment 3: Comm/Compute Ratio vs Matrix Size
    // =====================================================================
    printf("\n===== Exp 3: Comm/Compute Ratio vs Matrix Size (%d GPUs) =====\n", max_gpus);
    printf("%-6s  %10s  %10s  %8s\n", "Size", "GEMM(ms)", "Comm(ms)", "Ratio");
    printf("------------------------------------------------------\n");

    for (int S : sizes) {
        const int M = S, N = S, K = S;
        const int N_local = N / max_gpus;

        std::vector<ColParallelBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++)
            bufs.emplace_back(g, M, K, N_local, N);

        double gemm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                          M, N_local, K, ctx.compute_stream);
            ctx.compute_stream.synchronize();
        });

        double comm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local, ncclFloat,
                                     comms.get(max_gpus, g), ctx.comm_stream));
            ctx.comm_stream.synchronize();
        });

        printf("%-6d  %10.3f  %10.3f  %8.2f\n", S, gemm_ms, comm_ms, comm_ms / gemm_ms);
    }

    // =====================================================================
    // Experiment 4: Comm/Compute Ratio across Different Kernels
    // =====================================================================
    printf("\n===== Exp 4: Comm/Compute Ratio across Kernels (size=4096, %d GPUs) =====\n",
           max_gpus);
    printf("%-20s  %10s  %10s  %8s\n", "Kernel", "GEMM(ms)", "Comm(ms)", "Ratio");
    printf("------------------------------------------------------\n");

    {
        constexpr int S = 4096;
        const int M = S, N = S, K = S;
        const int N_local = N / max_gpus;

        std::vector<ColParallelBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++)
            bufs.emplace_back(g, M, K, N_local, N);

        double comm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local, ncclFloat,
                                     comms.get(max_gpus, g), ctx.comm_stream));
            ctx.comm_stream.synchronize();
        });

        for (const auto& kptr : KernelRegistry::all()) {
            const GemmKernel& test_kernel = *kptr;

            double gemm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                ctx.activate();

                test_kernel.launch(bufs[g].X.get(), bufs[g].W.get(),
                                   bufs[g].Y.get(), M, N_local, K,
                                   ctx.compute_stream);
                ctx.compute_stream.synchronize();
            });

            printf("%-20s  %10.3f  %10.3f  %8.2f\n", test_kernel.name(), gemm_ms,
                   comm_ms, comm_ms / gemm_ms);
        }
    }

    // =====================================================================
    // Experiment 5: Parallel MLP Forward + Backward
    // =====================================================================
    printf("\n===== Exp 5: Parallel MLP Forward + Backward (%d GPUs) =====\n", max_gpus);
    printf("%-6s %-6s %-6s %-6s  %10s  %10s  %10s\n", "M", "H", "N", "GPUs", "Fwd(ms)", "Bwd(ms)",
           "Total(ms)");
    printf("-----------------------------------------------------------\n");

    for (int S : sizes) {
        const int M = S, K = S, H = S, N = S;
        const int H_local = H / max_gpus;

        std::vector<MLPBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++)
            bufs.emplace_back(g, M, K, H_local, N);

        double fwd_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            parallel_mlp_forward(bufs[g].X.get(), bufs[g].W1.get(), bufs[g].W2.get(),
                                 bufs[g].Hidden.get(), bufs[g].YPartial.get(), bufs[g].Y.get(), M,
                                 K, H, N, max_gpus, g, ctx.handle, comms.get(max_gpus, g),
                                 ctx.compute_stream, kernel);
            ctx.compute_stream.synchronize();
        });

        double bwd_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            parallel_mlp_backward(bufs[g].X.get(), bufs[g].W1.get(), bufs[g].W2.get(),
                                  bufs[g].Hidden.get(), bufs[g].dY.get(), bufs[g].dW1.get(),
                                  bufs[g].dW2.get(), bufs[g].dHidden.get(),
                                  bufs[g].dXPartial.get(), bufs[g].dX.get(), M, K, H, N, max_gpus,
                                  g, ctx.handle, comms.get(max_gpus, g), ctx.compute_stream,
                                  kernel);
            ctx.compute_stream.synchronize();
        });

        printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %10.3f\n", M, H, N, max_gpus, fwd_ms,
               bwd_ms, fwd_ms + bwd_ms);
    }

    // =====================================================================
    // Experiment 6: Communication-Compute Overlap
    // =====================================================================
    printf("\n===== Exp 6: Row Parallel — No Overlap vs Overlap (%d GPUs) =====\n", max_gpus);
    printf("%-6s  %-8s  %10s  %10s  %8s\n", "Size", "Chunks", "NoOvlp(ms)", "Overlap(ms)",
           "Speedup");
    printf("------------------------------------------------------\n");

    for (int S : sizes) {
        const int M = S, N = S, K = S;
        const int K_local = K / max_gpus;
        constexpr int num_chunks = 4;

        std::vector<RowParallelBuffers> bufs;
        bufs.reserve(max_gpus);
        for (int g = 0; g < max_gpus; g++)
            bufs.emplace_back(g, M, K_local, N);

        double no_overlap_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            row_parallel_forward(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                                 bufs[g].YReduced.get(), M, N, K, max_gpus, g, ctx.handle,
                                 comms.get(max_gpus, g), ctx.compute_stream, kernel);
            ctx.compute_stream.synchronize();
        });

        double overlap_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            ctx.activate();
            row_parallel_forward_overlap(bufs[g].X.get(), bufs[g].W.get(), bufs[g].YOverlap.get(),
                                         bufs[g].YReducedOverlap.get(), M, N, K, max_gpus, g,
                                         num_chunks, ctx.handle, comms.get(max_gpus, g),
                                         ctx.compute_stream, ctx.comm_stream, kernel);
            ctx.compute_stream.synchronize();
            ctx.comm_stream.synchronize();
        });

        printf("%-6d  %-8d  %10.3f  %10.3f  %8.2fx\n", S, num_chunks, no_overlap_ms, overlap_ms,
               no_overlap_ms / overlap_ms);
    }

    printf("\nDone.\n");
    return 0;
}
