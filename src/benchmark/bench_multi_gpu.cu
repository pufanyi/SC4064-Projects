/*
 * Multi-GPU Tensor Parallelism Benchmark
 * ======================================
 * Runs true multi-rank NCCL experiments from a single process by launching
 * one host thread per active GPU. This keeps the benchmark self-contained
 * while ensuring every rank participates in collectives.
 *
 * Usage: ./bench_multi_gpu [max_num_gpus] [kernel_id]
 *   kernel_id: 0=naive, 1=coalesced, 2=smem, 3=1d, 4=2d,
 *              5=vectorized, 6=warptile, 7=cuBLAS (default)
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <thread>
#include <vector>

#include "../kernels/gemm_dispatch.cuh"
#include "../utils/cuda_utils.cuh"

#define NCCL_CHECK(cmd)                                                                            \
    do {                                                                                           \
        ncclResult_t r = cmd;                                                                      \
        if (r != ncclSuccess) {                                                                    \
            fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// ============================================================================
// Forward declarations from tensor_parallel.cu
// ============================================================================
void column_parallel_forward(const float* d_X, const float* d_W_shard, float* d_Y_local,
                             float* d_Y_full, int M, int N, int K, int num_gpus, int gpu_id,
                             cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                             GemmKernel kernel);

void row_parallel_forward(const float* d_X_shard, const float* d_W_shard, float* d_Y_local,
                          float* d_Y_reduced, int M, int N, int K, int num_gpus, int gpu_id,
                          cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                          GemmKernel kernel);

void parallel_mlp_forward(const float* d_X, const float* d_W1_shard, const float* d_W2_shard,
                          float* d_hidden, float* d_Y_partial, float* d_Y, int M, int K, int H,
                          int N, int num_gpus, int gpu_id, cublasHandle_t handle, ncclComm_t comm,
                          cudaStream_t stream, GemmKernel kernel);

void parallel_mlp_backward(const float* d_X, const float* d_W1_shard, const float* d_W2_shard,
                           const float* d_hidden, const float* d_dY, float* d_dW1_shard,
                           float* d_dW2_shard, float* d_d_hidden, float* d_dX_partial, float* d_dX,
                           int M, int K, int H, int N, int num_gpus, int gpu_id,
                           cublasHandle_t handle, ncclComm_t comm, cudaStream_t stream,
                           GemmKernel kernel);

void row_parallel_forward_overlap(const float* d_X_shard, const float* d_W_shard, float* d_Y_local,
                                  float* d_Y_reduced, int M, int N, int K, int num_gpus, int gpu_id,
                                  int num_chunks, cublasHandle_t handle, ncclComm_t comm,
                                  cudaStream_t compute_stream, cudaStream_t comm_stream,
                                  GemmKernel kernel);

// ============================================================================
// Small helpers
// ============================================================================

struct DeviceContext {
    int device_id = -1;
    cublasHandle_t handle = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t comm_stream = nullptr;
};

struct CommRegistry {
    std::map<int, std::vector<ncclComm_t>> groups;

    explicit CommRegistry(int max_gpus) {
        std::vector<int> devs(max_gpus);
        for (int i = 0; i < max_gpus; i++) devs[i] = i;

        for (int p = 2; p <= max_gpus; p *= 2) {
            groups[p].resize(p);
            NCCL_CHECK(ncclCommInitAll(groups[p].data(), p, devs.data()));
        }
        if (max_gpus > 1 && (max_gpus & (max_gpus - 1)) != 0) {
            groups[max_gpus].resize(max_gpus);
            NCCL_CHECK(ncclCommInitAll(groups[max_gpus].data(), max_gpus, devs.data()));
        }
    }

    ~CommRegistry() {
        for (auto& entry : groups) {
            for (ncclComm_t comm : entry.second) {
                if (comm) ncclCommDestroy(comm);
            }
        }
    }

    ncclComm_t get(int active_gpus, int gpu_id) const {
        if (active_gpus <= 1) return nullptr;
        return groups.at(active_gpus)[gpu_id];
    }
};

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

void init_device_matrix(int device_id, float* d_ptr, int rows, int cols, unsigned seed) {
    std::vector<float> host(rows * cols);
    init_matrix(host.data(), rows, cols, seed);
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
}

template <typename T, typename Fn>
void free_device_buffers(std::vector<T>& buffers, int active_gpus, Fn free_fn) {
    for (int g = 0; g < active_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        free_fn(buffers[g]);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    int max_gpus = 2;
    GemmKernel kernel = GemmKernel::CUBLAS;

    if (argc > 1) max_gpus = atoi(argv[1]);
    if (argc > 2) {
        int kid = atoi(argv[2]);
        if (kid >= 0 && kid < static_cast<int>(GemmKernel::NUM_KERNELS)) {
            kernel = static_cast<GemmKernel>(kid);
        }
    }

    int available_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&available_gpus));
    if (available_gpus <= 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }
    max_gpus = std::min(max_gpus, available_gpus);

    printf("===== Multi-GPU Tensor Parallel Benchmark =====\n");
    printf("Max GPUs used:      %d\n", max_gpus);
    printf("Local GEMM kernel:  %s\n\n", gemm_kernel_name(kernel));

    std::vector<DeviceContext> contexts(max_gpus);
    for (int g = 0; g < max_gpus; g++) {
        contexts[g].device_id = g;
        CUDA_CHECK(cudaSetDevice(g));
        print_device_info();
        CUBLAS_CHECK(cublasCreate(&contexts[g].handle));
        CUDA_CHECK(cudaStreamCreate(&contexts[g].compute_stream));
        CUDA_CHECK(cudaStreamCreate(&contexts[g].comm_stream));
    }

    CommRegistry comms(max_gpus);
    const std::vector<int> scaling_counts = build_scaling_counts(max_gpus);
    const std::vector<int> sizes = {2048, 4096, 8192};
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
            struct Buffers {
                float* dX = nullptr;
                float* dW = nullptr;
                float* dY = nullptr;
                float* dY_full = nullptr;
            };
            std::vector<Buffers> bufs(active_gpus);
            const int M = S, N = S, K = S;
            const int N_local = N / active_gpus;

            for (int g = 0; g < active_gpus; g++) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaMalloc(&bufs[g].dX, M * K * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&bufs[g].dW, K * N_local * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&bufs[g].dY, M * N_local * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&bufs[g].dY_full, M * N * sizeof(float)));
                init_device_matrix(g, bufs[g].dX, M, K, 42);
                init_device_matrix(g, bufs[g].dW, K, N_local, 137 + g);
            }

            double gemm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                CUDA_CHECK(cudaSetDevice(ctx.device_id));
                dispatch_gemm_on_stream(kernel, bufs[g].dX, bufs[g].dW, bufs[g].dY, M, N_local, K,
                                        ctx.handle, ctx.compute_stream);
                CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
            });

            double comm_ms = 0.0;
            if (active_gpus > 1) {
                comm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                    auto& ctx = contexts[g];
                    CUDA_CHECK(cudaSetDevice(ctx.device_id));
                    NCCL_CHECK(ncclAllGather(bufs[g].dY, bufs[g].dY_full, M * N_local, ncclFloat,
                                             comms.get(active_gpus, g), ctx.comm_stream));
                    CUDA_CHECK(cudaStreamSynchronize(ctx.comm_stream));
                });
            }

            double total_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                CUDA_CHECK(cudaSetDevice(ctx.device_id));
                column_parallel_forward(bufs[g].dX, bufs[g].dW, bufs[g].dY, bufs[g].dY_full, M, N,
                                        K, active_gpus, g, ctx.handle, comms.get(active_gpus, g),
                                        ctx.compute_stream, kernel);
                CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
            });

            printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %10.3f  %8.1f\n", M, N, K, active_gpus,
                   gemm_ms, comm_ms, total_ms, gemm_gflops(M, N, K, total_ms));

            free_device_buffers(bufs, active_gpus, [](Buffers& buf) {
                CUDA_CHECK(cudaFree(buf.dX));
                CUDA_CHECK(cudaFree(buf.dW));
                CUDA_CHECK(cudaFree(buf.dY));
                CUDA_CHECK(cudaFree(buf.dY_full));
            });
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
        struct Buffers {
            float* dX = nullptr;
            float* dW = nullptr;
            float* dY = nullptr;
            float* dY_full = nullptr;
        };
        std::vector<Buffers> bufs(active_gpus);
        const int M = 2048, K = 2048, N_local = 2048;
        const int N_total = N_local * active_gpus;

        for (int g = 0; g < active_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&bufs[g].dX, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dW, K * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY, M * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY_full, M * N_total * sizeof(float)));
            init_device_matrix(g, bufs[g].dX, M, K, 42);
            init_device_matrix(g, bufs[g].dW, K, N_local, 137 + g);
        }

        double gemm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            dispatch_gemm_on_stream(kernel, bufs[g].dX, bufs[g].dW, bufs[g].dY, M, N_local, K,
                                    ctx.handle, ctx.compute_stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
        });

        double comm_ms = 0.0;
        if (active_gpus > 1) {
            comm_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                CUDA_CHECK(cudaSetDevice(ctx.device_id));
                NCCL_CHECK(ncclAllGather(bufs[g].dY, bufs[g].dY_full, M * N_local, ncclFloat,
                                         comms.get(active_gpus, g), ctx.comm_stream));
                CUDA_CHECK(cudaStreamSynchronize(ctx.comm_stream));
            });
        }

        double total_ms = benchmark_wall_ms(active_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            column_parallel_forward(bufs[g].dX, bufs[g].dW, bufs[g].dY, bufs[g].dY_full, M, N_total,
                                    K, active_gpus, g, ctx.handle, comms.get(active_gpus, g),
                                    ctx.compute_stream, kernel);
            CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
        });

        printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %10.3f  %8.1f\n", M, N_total, K, active_gpus,
               gemm_ms, comm_ms, total_ms, gemm_gflops(M, N_total, K, total_ms));

        free_device_buffers(bufs, active_gpus, [](Buffers& buf) {
            CUDA_CHECK(cudaFree(buf.dX));
            CUDA_CHECK(cudaFree(buf.dW));
            CUDA_CHECK(cudaFree(buf.dY));
            CUDA_CHECK(cudaFree(buf.dY_full));
        });
    }

    // =====================================================================
    // Experiment 3: Comm/Compute Ratio vs Matrix Size
    // =====================================================================
    printf("\n===== Exp 3: Comm/Compute Ratio vs Matrix Size (%d GPUs) =====\n", max_gpus);
    printf("%-6s  %10s  %10s  %8s\n", "Size", "GEMM(ms)", "Comm(ms)", "Ratio");
    printf("------------------------------------------------------\n");

    for (int S : sizes) {
        struct Buffers {
            float* dX = nullptr;
            float* dW = nullptr;
            float* dY = nullptr;
            float* dY_full = nullptr;
        };
        std::vector<Buffers> bufs(max_gpus);
        const int M = S, N = S, K = S;
        const int N_local = N / max_gpus;

        for (int g = 0; g < max_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&bufs[g].dX, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dW, K * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY, M * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY_full, M * N * sizeof(float)));
            init_device_matrix(g, bufs[g].dX, M, K, 42);
            init_device_matrix(g, bufs[g].dW, K, N_local, 137 + g);
        }

        double gemm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            dispatch_gemm_on_stream(kernel, bufs[g].dX, bufs[g].dW, bufs[g].dY, M, N_local, K,
                                    ctx.handle, ctx.compute_stream);
            CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
        });

        double comm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            NCCL_CHECK(ncclAllGather(bufs[g].dY, bufs[g].dY_full, M * N_local, ncclFloat,
                                     comms.get(max_gpus, g), ctx.comm_stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx.comm_stream));
        });

        printf("%-6d  %10.3f  %10.3f  %8.2f\n", S, gemm_ms, comm_ms, comm_ms / gemm_ms);

        free_device_buffers(bufs, max_gpus, [](Buffers& buf) {
            CUDA_CHECK(cudaFree(buf.dX));
            CUDA_CHECK(cudaFree(buf.dW));
            CUDA_CHECK(cudaFree(buf.dY));
            CUDA_CHECK(cudaFree(buf.dY_full));
        });
    }

    // =====================================================================
    // Experiment 4: Comm/Compute Ratio across Different Kernels
    // =====================================================================
    printf("\n===== Exp 4: Comm/Compute Ratio across Kernels (size=4096, %d GPUs) =====\n",
           max_gpus);
    printf("%-20s  %10s  %10s  %8s\n", "Kernel", "GEMM(ms)", "Comm(ms)", "Ratio");
    printf("------------------------------------------------------\n");

    {
        struct Buffers {
            float* dX = nullptr;
            float* dW = nullptr;
            float* dY = nullptr;
            float* dY_full = nullptr;
        };
        std::vector<Buffers> bufs(max_gpus);
        const int S = 4096;
        const int M = S, N = S, K = S;
        const int N_local = N / max_gpus;

        for (int g = 0; g < max_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&bufs[g].dX, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dW, K * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY, M * N_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY_full, M * N * sizeof(float)));
            init_device_matrix(g, bufs[g].dX, M, K, 42);
            init_device_matrix(g, bufs[g].dW, K, N_local, 137 + g);
        }

        double comm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            NCCL_CHECK(ncclAllGather(bufs[g].dY, bufs[g].dY_full, M * N_local, ncclFloat,
                                     comms.get(max_gpus, g), ctx.comm_stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx.comm_stream));
        });

        const std::vector<GemmKernel> test_kernels = {
            GemmKernel::NAIVE,        GemmKernel::COALESCED,    GemmKernel::SMEM,
            GemmKernel::BLOCKTILE_1D, GemmKernel::BLOCKTILE_2D, GemmKernel::VECTORIZED,
            GemmKernel::WARPTILE,     GemmKernel::CUBLAS};

        for (GemmKernel test_kernel : test_kernels) {
            double gemm_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
                auto& ctx = contexts[g];
                CUDA_CHECK(cudaSetDevice(ctx.device_id));
                dispatch_gemm_on_stream(test_kernel, bufs[g].dX, bufs[g].dW, bufs[g].dY, M, N_local,
                                        K, ctx.handle, ctx.compute_stream);
                CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
            });

            printf("%-20s  %10.3f  %10.3f  %8.2f\n", gemm_kernel_name(test_kernel), gemm_ms,
                   comm_ms, comm_ms / gemm_ms);
        }

        free_device_buffers(bufs, max_gpus, [](Buffers& buf) {
            CUDA_CHECK(cudaFree(buf.dX));
            CUDA_CHECK(cudaFree(buf.dW));
            CUDA_CHECK(cudaFree(buf.dY));
            CUDA_CHECK(cudaFree(buf.dY_full));
        });
    }

    // =====================================================================
    // Experiment 5: Parallel MLP Forward + Backward
    // =====================================================================
    printf("\n===== Exp 5: Parallel MLP Forward + Backward (%d GPUs) =====\n", max_gpus);
    printf("%-6s %-6s %-6s %-6s  %10s  %10s  %10s\n", "M", "H", "N", "GPUs", "Fwd(ms)", "Bwd(ms)",
           "Total(ms)");
    printf("-----------------------------------------------------------\n");

    for (int S : sizes) {
        struct Buffers {
            float* dX = nullptr;
            float* dW1 = nullptr;
            float* dW2 = nullptr;
            float* dHidden = nullptr;
            float* dYPartial = nullptr;
            float* dY = nullptr;
            float* dDY = nullptr;
            float* dDW1 = nullptr;
            float* dDW2 = nullptr;
            float* dDHidden = nullptr;
            float* dDXPartial = nullptr;
            float* dDX = nullptr;
        };
        std::vector<Buffers> bufs(max_gpus);
        const int M = S, K = S, H = S, N = S;
        const int H_local = H / max_gpus;

        for (int g = 0; g < max_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&bufs[g].dX, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dW1, K * H_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dW2, H_local * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dHidden, M * H_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dYPartial, M * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY, M * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dDY, M * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dDW1, K * H_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dDW2, H_local * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dDHidden, M * H_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dDXPartial, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dDX, M * K * sizeof(float)));

            init_device_matrix(g, bufs[g].dX, M, K, 42);
            init_device_matrix(g, bufs[g].dW1, K, H_local, 137 + g);
            init_device_matrix(g, bufs[g].dW2, H_local, N, 271 + g);
            init_device_matrix(g, bufs[g].dDY, M, N, 314 + g);
        }

        double fwd_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            parallel_mlp_forward(bufs[g].dX, bufs[g].dW1, bufs[g].dW2, bufs[g].dHidden,
                                 bufs[g].dYPartial, bufs[g].dY, M, K, H, N, max_gpus, g, ctx.handle,
                                 comms.get(max_gpus, g), ctx.compute_stream, kernel);
            CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
        });

        double bwd_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            parallel_mlp_backward(bufs[g].dX, bufs[g].dW1, bufs[g].dW2, bufs[g].dHidden,
                                  bufs[g].dDY, bufs[g].dDW1, bufs[g].dDW2, bufs[g].dDHidden,
                                  bufs[g].dDXPartial, bufs[g].dDX, M, K, H, N, max_gpus, g,
                                  ctx.handle, comms.get(max_gpus, g), ctx.compute_stream, kernel);
            CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
        });

        printf("%-6d %-6d %-6d %-6d  %10.3f  %10.3f  %10.3f\n", M, H, N, max_gpus, fwd_ms, bwd_ms,
               fwd_ms + bwd_ms);

        free_device_buffers(bufs, max_gpus, [](Buffers& buf) {
            CUDA_CHECK(cudaFree(buf.dX));
            CUDA_CHECK(cudaFree(buf.dW1));
            CUDA_CHECK(cudaFree(buf.dW2));
            CUDA_CHECK(cudaFree(buf.dHidden));
            CUDA_CHECK(cudaFree(buf.dYPartial));
            CUDA_CHECK(cudaFree(buf.dY));
            CUDA_CHECK(cudaFree(buf.dDY));
            CUDA_CHECK(cudaFree(buf.dDW1));
            CUDA_CHECK(cudaFree(buf.dDW2));
            CUDA_CHECK(cudaFree(buf.dDHidden));
            CUDA_CHECK(cudaFree(buf.dDXPartial));
            CUDA_CHECK(cudaFree(buf.dDX));
        });
    }

    // =====================================================================
    // Experiment 6: Communication-Compute Overlap
    // =====================================================================
    printf("\n===== Exp 6: Row Parallel — No Overlap vs Overlap (%d GPUs) =====\n", max_gpus);
    printf("%-6s  %-8s  %10s  %10s  %8s\n", "Size", "Chunks", "NoOvlp(ms)", "Overlap(ms)",
           "Speedup");
    printf("------------------------------------------------------\n");

    for (int S : sizes) {
        struct Buffers {
            float* dX = nullptr;
            float* dW = nullptr;
            float* dY = nullptr;
            float* dYReduced = nullptr;
            float* dYOverlap = nullptr;
            float* dYReducedOverlap = nullptr;
        };
        std::vector<Buffers> bufs(max_gpus);
        const int M = S, N = S, K = S;
        const int K_local = K / max_gpus;
        const int num_chunks = 4;

        for (int g = 0; g < max_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&bufs[g].dX, M * K_local * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dW, K_local * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dY, M * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dYReduced, M * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dYOverlap, M * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&bufs[g].dYReducedOverlap, M * N * sizeof(float)));
            init_device_matrix(g, bufs[g].dX, M, K_local, 42);
            init_device_matrix(g, bufs[g].dW, K_local, N, 137 + g);
        }

        double no_overlap_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            row_parallel_forward(bufs[g].dX, bufs[g].dW, bufs[g].dY, bufs[g].dYReduced, M, N, K,
                                 max_gpus, g, ctx.handle, comms.get(max_gpus, g),
                                 ctx.compute_stream, kernel);
            CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
        });

        double overlap_ms = benchmark_wall_ms(max_gpus, kWarmup, kRepeat, [&](int g) {
            auto& ctx = contexts[g];
            CUDA_CHECK(cudaSetDevice(ctx.device_id));
            row_parallel_forward_overlap(bufs[g].dX, bufs[g].dW, bufs[g].dYOverlap,
                                         bufs[g].dYReducedOverlap, M, N, K, max_gpus, g, num_chunks,
                                         ctx.handle, comms.get(max_gpus, g), ctx.compute_stream,
                                         ctx.comm_stream, kernel);
            CUDA_CHECK(cudaStreamSynchronize(ctx.compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(ctx.comm_stream));
        });

        printf("%-6d  %-8d  %10.3f  %10.3f  %8.2fx\n", S, num_chunks, no_overlap_ms, overlap_ms,
               no_overlap_ms / overlap_ms);

        free_device_buffers(bufs, max_gpus, [](Buffers& buf) {
            CUDA_CHECK(cudaFree(buf.dX));
            CUDA_CHECK(cudaFree(buf.dW));
            CUDA_CHECK(cudaFree(buf.dY));
            CUDA_CHECK(cudaFree(buf.dYReduced));
            CUDA_CHECK(cudaFree(buf.dYOverlap));
            CUDA_CHECK(cudaFree(buf.dYReducedOverlap));
        });
    }

    for (int g = 0; g < max_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        cublasDestroy(contexts[g].handle);
        CUDA_CHECK(cudaStreamDestroy(contexts[g].compute_stream));
        CUDA_CHECK(cudaStreamDestroy(contexts[g].comm_stream));
    }

    printf("\nDone.\n");
    return 0;
}
