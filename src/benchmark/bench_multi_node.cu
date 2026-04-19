// Copyright 2025 Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au
// SC4064 GPU Programming, Nanyang Technological University
//
// bench_multi_node.cu - Cross-node tensor parallel benchmark.
//
// Process model:   ONE process per node, N threads per process (one per local
//                  GPU).  No MPI required.
// Coordination:    PyTorch / K8s PyTorchJob env convention --
//                    MASTER_ADDR      hostname of node_rank 0
//                    MASTER_PORT      TCP port used for NCCL-id rendezvous
//                    WORLD_SIZE       number of NODES (not ranks)
//                    RANK             this node's rank, 0..WORLD_SIZE-1
//                  Node 0 calls ncclGetUniqueId, opens a TCP listener on
//                  MASTER_PORT, sends the id to each connecting peer.  Other
//                  nodes connect and recv.  All threads then call
//                  ncclCommInitRank(global_rank = node_rank * local_gpus + i).
//
// Experiments:     Same 6 experiments as bench_multi_gpu.cu, but only at the
//                  global world size (num_nodes * local_gpus).  The 1..8 GPU
//                  scaling points come from the single-node bench already.

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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
// Environment / bootstrap
// ============================================================================

struct NodeEnv {
    int node_rank = 0;
    int num_nodes = 1;
    int local_gpus = 1;
    int world_size = 1;
    std::string master_addr = "localhost";
    int master_port = 23456;
};

static const char* env_or(const char* name, const char* fallback) {
    const char* v = getenv(name);
    return (v && *v) ? v : fallback;
}

static NodeEnv read_env() {
    NodeEnv e;
    e.node_rank = atoi(env_or("RANK", "0"));
    e.num_nodes = atoi(env_or("WORLD_SIZE", "1"));
    e.master_addr = env_or("MASTER_ADDR", "localhost");
    e.master_port = atoi(env_or("MASTER_PORT", "23456"));
    CUDA_CHECK(cudaGetDeviceCount(&e.local_gpus));
    e.world_size = e.num_nodes * e.local_gpus;
    return e;
}

// --- TCP send/recv full buffer (loop until complete) -----------------------
static void send_all(int fd, const void* buf, size_t n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        ssize_t s = send(fd, p, n, 0);
        if (s <= 0) {
            perror("send");
            exit(EXIT_FAILURE);
        }
        p += s;
        n -= s;
    }
}

static void recv_all(int fd, void* buf, size_t n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) {
            perror("recv");
            exit(EXIT_FAILURE);
        }
        p += r;
        n -= r;
    }
}

// --- Broadcast NCCL unique id from node 0 to all other nodes over TCP ------
static void tcp_broadcast_nccl_id(ncclUniqueId& id, const NodeEnv& env) {
    if (env.num_nodes == 1) {
        NCCL_CHECK(ncclGetUniqueId(&id));
        return;
    }

    if (env.node_rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));

        int srv = socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) {
            perror("socket");
            exit(EXIT_FAILURE);
        }
        int yes = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in sa{};
        sa.sin_family = AF_INET;
        sa.sin_addr.s_addr = INADDR_ANY;
        sa.sin_port = htons(env.master_port);
        if (bind(srv, reinterpret_cast<sockaddr*>(&sa), sizeof(sa)) != 0) {
            perror("bind");
            exit(EXIT_FAILURE);
        }
        if (listen(srv, env.num_nodes) != 0) {
            perror("listen");
            exit(EXIT_FAILURE);
        }

        printf("[node 0] listening on port %d for %d peer(s)...\n", env.master_port,
               env.num_nodes - 1);
        fflush(stdout);

        for (int i = 1; i < env.num_nodes; i++) {
            sockaddr_in ca{};
            socklen_t clen = sizeof(ca);
            int c = accept(srv, reinterpret_cast<sockaddr*>(&ca), &clen);
            if (c < 0) {
                perror("accept");
                exit(EXIT_FAILURE);
            }
            char ip[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, &ca.sin_addr, ip, sizeof(ip));
            printf("[node 0] peer %d connected from %s\n", i, ip);
            fflush(stdout);
            send_all(c, &id, sizeof(id));
            close(c);
        }
        close(srv);
    } else {
        char port_s[16];
        snprintf(port_s, sizeof(port_s), "%d", env.master_port);

        int s = -1;
        // Up to 1 hour of 1-second retries.  Master may be slow starting (e.g.
        // still building, or running its own local stages); we want the worker
        // to patiently wait rather than fail fast.
        for (int attempt = 0; attempt < 3600; attempt++) {
            addrinfo hints{};
            addrinfo* res = nullptr;
            hints.ai_family = AF_UNSPEC;
            hints.ai_socktype = SOCK_STREAM;
            int rc = getaddrinfo(env.master_addr.c_str(), port_s, &hints, &res);
            if (rc != 0 || !res) {
                if (attempt == 0)
                    fprintf(stderr, "[node %d] getaddrinfo(%s) failed, retrying...\n",
                            env.node_rank, env.master_addr.c_str());
                sleep(1);
                continue;
            }
            for (addrinfo* rp = res; rp; rp = rp->ai_next) {
                s = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
                if (s < 0) continue;
                if (connect(s, rp->ai_addr, rp->ai_addrlen) == 0) {
                    freeaddrinfo(res);
                    goto connected;
                }
                close(s);
                s = -1;
            }
            freeaddrinfo(res);
            sleep(1);
        }
    connected:
        if (s < 0) {
            fprintf(stderr, "[node %d] could not connect to %s:%d after retries\n", env.node_rank,
                    env.master_addr.c_str(), env.master_port);
            exit(EXIT_FAILURE);
        }
        printf("[node %d] connected to master %s:%d\n", env.node_rank, env.master_addr.c_str(),
               env.master_port);
        fflush(stdout);
        recv_all(s, &id, sizeof(id));
        close(s);
    }
}

// ============================================================================
// Per-GPU context (owned by each worker thread)
// ============================================================================
struct DeviceContext {
    int local_gpu;
    int world_rank;
    ncclComm_t comm = nullptr;
    CublasHandle handle;        // created lazily on thread's device
    CudaStream compute_stream;  // created lazily
    CudaStream comm_stream;

    DeviceContext() = default;
    DeviceContext(const DeviceContext&) = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
};

// Build all per-GPU NCCL communicators inside one ncclGroupStart/End call.
//
// IMPORTANT: cudaSetDevice(i) MUST run before DeviceContext is constructed --
// CudaStream and CublasHandle are created in the member-init list, which
// happens before the constructor body, so the current device at the moment
// of `make_unique<DeviceContext>()` is what the streams / handle bind to.
// Getting this wrong produces streams bound to the *previous* GPU, which
// works accidentally for IB/RDMA transports (NIC bypasses the stream) but
// surfaces as "unhandled cuda error" for TCP-socket / Ring-kernel paths
// that rely on the stream executing on the comm's device.
static void init_device_contexts(std::vector<std::unique_ptr<DeviceContext>>& ctxs,
                                 const NodeEnv& env, const ncclUniqueId& id) {
    ctxs.resize(env.local_gpus);
    for (int i = 0; i < env.local_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        ctxs[i] = std::make_unique<DeviceContext>();
        ctxs[i]->local_gpu = i;
        ctxs[i]->world_rank = env.node_rank * env.local_gpus + i;
    }

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < env.local_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        NCCL_CHECK(ncclCommInitRank(&ctxs[i]->comm, env.world_size, id, ctxs[i]->world_rank));
    }
    NCCL_CHECK(ncclGroupEnd());
}

// Activate a per-GPU context on the current thread (sets device + cuBLAS handle
// for any kernel that needs one).
static void activate(DeviceContext& ctx) {
    CUDA_CHECK(cudaSetDevice(ctx.local_gpu));
    for (int i = 0; i < KernelRegistry::count(); i++) {
        auto& k = KernelRegistry::get_mut(i);
        if (k.needs_cublas()) k.set_cublas_handle(ctx.handle);
    }
}

// ============================================================================
// Multi-threaded execution helpers (same pattern as bench_multi_gpu.cu)
// ============================================================================

template <typename Fn>
static void run_on_local_gpus(int local_gpus, Fn fn) {
    std::vector<std::thread> workers;
    workers.reserve(local_gpus);
    for (int g = 0; g < local_gpus; g++) workers.emplace_back([&, g]() { fn(g); });
    for (auto& w : workers) w.join();
}

template <typename Fn>
static BenchStats benchmark_stats(int local_gpus, int warmup, int repeat, Fn fn) {
    for (int i = 0; i < warmup; i++) run_on_local_gpus(local_gpus, fn);

    std::vector<double> samples;
    samples.reserve(repeat);
    for (int i = 0; i < repeat; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        run_on_local_gpus(local_gpus, fn);
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return compute_stats(samples);
}

// Cross-node barrier: issue a small NCCL allreduce on every local GPU so every
// node waits for every other node before proceeding.  Cheap relative to
// experiment ops; we use it to align node clocks before a measurement.
static void cross_node_barrier(std::vector<std::unique_ptr<DeviceContext>>& ctxs,
                               std::vector<CudaMemory<float>>& barrier_bufs) {
    const int n = static_cast<int>(ctxs.size());
    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < n; i++) {
        CUDA_CHECK(cudaSetDevice(ctxs[i]->local_gpu));
        NCCL_CHECK(ncclAllReduce(barrier_bufs[i].get(), barrier_bufs[i].get(), 1, ncclFloat,
                                 ncclSum, ctxs[i]->comm, ctxs[i]->comm_stream));
    }
    NCCL_CHECK(ncclGroupEnd());
    for (int i = 0; i < n; i++) {
        CUDA_CHECK(cudaSetDevice(ctxs[i]->local_gpu));
        ctxs[i]->comm_stream.synchronize();
    }
}

// ============================================================================
// Buffer structs (identical to bench_multi_gpu.cu)
// ============================================================================

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

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    NodeEnv env = read_env();

    // Kernel id (default = last registered, which is cuBLAS)
    int kernel_id = KernelRegistry::count() - 1;
    if (argc > 1) {
        int kid = atoi(argv[1]);
        if (kid >= 0 && kid < KernelRegistry::count()) kernel_id = kid;
    }
    const GemmKernel& kernel = KernelRegistry::get(kernel_id);

    const bool is_master = (env.node_rank == 0);

    if (is_master) {
        printf("===== Multi-Node Tensor Parallel Benchmark =====\n");
        printf("World size:         %d\n", env.world_size);
        printf("Nodes:              %d\n", env.num_nodes);
        printf("GPUs per node:      %d\n", env.local_gpus);
        printf("Master:             %s:%d\n", env.master_addr.c_str(), env.master_port);
        printf("Local GEMM kernel:  %s\n", kernel.name());
        printf("Transport tag:      %s\n", env_or("TRANSPORT_TAG", "default"));
        // Echo the NCCL env vars that actually select a transport -- helpful when
        // cross-checking which backend NCCL picked.
        const char* kNcclVars[] = {"NCCL_NET",           "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE",
                                   "NCCL_SOCKET_IFNAME", "NCCL_ALGO",       "NCCL_PROTO",
                                   "NCCL_NET_GDR_LEVEL", "NCCL_IB_HCA"};
        for (const char* k : kNcclVars) {
            const char* v = getenv(k);
            printf("  %-22s %s\n", k, (v && *v) ? v : "(unset)");
        }
        printf("\n");
        for (int g = 0; g < env.local_gpus; g++) {
            CUDA_CHECK(cudaSetDevice(g));
            print_device_info();
        }
        fflush(stdout);
    }

    // --- Bootstrap NCCL ---------------------------------------------------
    ncclUniqueId nccl_id;
    tcp_broadcast_nccl_id(nccl_id, env);

    std::vector<std::unique_ptr<DeviceContext>> ctxs;
    init_device_contexts(ctxs, env, nccl_id);

    // Per-GPU 1-element buffers for cross-node barriers
    std::vector<CudaMemory<float>> barrier_bufs;
    barrier_bufs.reserve(env.local_gpus);
    for (int g = 0; g < env.local_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        barrier_bufs.emplace_back(1);
        CUDA_CHECK(cudaMemset(barrier_bufs.back().get(), 0, sizeof(float)));
    }
    // Align the clock on every node before the first measurement.
    cross_node_barrier(ctxs, barrier_bufs);

    // Size lists tuned to the per-GPU memory budget at 16 GPUs x 80 GB.
    // In practice 49152 already OOMs on the Exp 1 allocation sweep once the
    // CUDA pool has cycled through the 32768 case (fragmentation + NCCL
    // workspace + GDRDMA regions push us past what's nominally free), so
    // we cap everything at 32768.
    //
    // Note: multi_node only runs Exp 1 + Exp 3 -- the cross-node strong-scaling
    // datapoint and the comm/compute ratio used by the transport sweep.
    // Exp 2/4/5/5b/6 already run at 1..8 GPUs (NVLink) in bench_multi_gpu,
    // which is the representative scale for those characterisations; redoing
    // them at 16 GPUs over IB would just have AllGather dominate the numbers.
    const std::vector<int> col_sizes = {2048, 4096, 8192, 16384, 32768};
    constexpr int kWarmup = 5;
    constexpr int kRepeat = 20;

    // =====================================================================
    // Experiment 1: Strong Scaling — Column Parallel Forward (at world_size)
    // =====================================================================
    if (is_master) {
        printf("===== Exp 1: Strong Scaling — Column Parallel Forward =====\n");
        printf("%-6s %-6s %-6s %-6s  %10s %10s  %10s %10s  %10s %10s  %8s\n", "M", "N", "K", "GPUs",
               "GEMM(ms)", "GEMM_std", "Comm(ms)", "Comm_std", "Total(ms)", "Total_std", "GFLOPS");
        printf(
            "--------------------------------------------------------------------------"
            "-----------------------\n");
        fflush(stdout);
    }

    for (int S : col_sizes) {
        const int M = S, N = S, K = S;
        const int N_local = N / env.world_size;

        std::vector<ColParallelBuffers> bufs;
        bufs.reserve(env.local_gpus);
        for (int g = 0; g < env.local_gpus; g++) bufs.emplace_back(g, M, K, N_local, N);

        cross_node_barrier(ctxs, barrier_bufs);
        BenchStats gemm = benchmark_stats(env.local_gpus, kWarmup, kRepeat, [&](int g) {
            activate(*ctxs[g]);
            kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(), M, N_local, K,
                          ctxs[g]->compute_stream);
            ctxs[g]->compute_stream.synchronize();
        });

        cross_node_barrier(ctxs, barrier_bufs);
        BenchStats comm = benchmark_stats(env.local_gpus, kWarmup, kRepeat, [&](int g) {
            activate(*ctxs[g]);
            NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local, ncclFloat,
                                     ctxs[g]->comm, ctxs[g]->comm_stream));
            ctxs[g]->comm_stream.synchronize();
        });

        cross_node_barrier(ctxs, barrier_bufs);
        BenchStats total = benchmark_stats(env.local_gpus, kWarmup, kRepeat, [&](int g) {
            activate(*ctxs[g]);
            column_parallel_forward(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(),
                                    bufs[g].Y_full.get(), M, N, K, env.world_size,
                                    ctxs[g]->world_rank, ctxs[g]->handle, ctxs[g]->comm,
                                    ctxs[g]->compute_stream, kernel);
            ctxs[g]->compute_stream.synchronize();
        });

        if (is_master) {
            printf("%-6d %-6d %-6d %-6d  %10.3f %10.3f  %10.3f %10.3f  %10.3f %10.3f  %8.1f\n", M,
                   N, K, env.world_size, gemm.mean, gemm.stddev, comm.mean, comm.stddev, total.mean,
                   total.stddev, gemm_gflops(M, N, K, total.mean));
            fflush(stdout);
        }
    }

    // Exp 2/4/5/5b/6 are intentionally not run here -- see comment at the
    // top of this function.  They live in bench_multi_gpu.

    // =====================================================================
    // Experiment 3: Comm/Compute Ratio vs Matrix Size (at world_size)
    // =====================================================================
    if (is_master) {
        printf("\n===== Exp 3: Comm/Compute Ratio vs Matrix Size (%d GPUs) =====\n",
               env.world_size);
        printf("%-6s  %10s %10s  %10s %10s  %8s\n", "Size", "GEMM(ms)", "GEMM_std", "Comm(ms)",
               "Comm_std", "Ratio");
        printf("----------------------------------------------------------------------\n");
        fflush(stdout);
    }

    for (int S : col_sizes) {
        const int M = S, N = S, K = S;
        const int N_local = N / env.world_size;

        std::vector<ColParallelBuffers> bufs;
        bufs.reserve(env.local_gpus);
        for (int g = 0; g < env.local_gpus; g++) bufs.emplace_back(g, M, K, N_local, N);

        cross_node_barrier(ctxs, barrier_bufs);
        BenchStats gemm = benchmark_stats(env.local_gpus, kWarmup, kRepeat, [&](int g) {
            activate(*ctxs[g]);
            kernel.launch(bufs[g].X.get(), bufs[g].W.get(), bufs[g].Y.get(), M, N_local, K,
                          ctxs[g]->compute_stream);
            ctxs[g]->compute_stream.synchronize();
        });

        cross_node_barrier(ctxs, barrier_bufs);
        BenchStats comm = benchmark_stats(env.local_gpus, kWarmup, kRepeat, [&](int g) {
            activate(*ctxs[g]);
            NCCL_CHECK(ncclAllGather(bufs[g].Y.get(), bufs[g].Y_full.get(), M * N_local, ncclFloat,
                                     ctxs[g]->comm, ctxs[g]->comm_stream));
            ctxs[g]->comm_stream.synchronize();
        });

        if (is_master) {
            printf("%-6d  %10.3f %10.3f  %10.3f %10.3f  %8.2f\n", S, gemm.mean, gemm.stddev,
                   comm.mean, comm.stddev, gemm.mean > 0 ? comm.mean / gemm.mean : 0.0);
            fflush(stdout);
        }
    }

    if (is_master) {
        printf("\nDone.\n");
        fflush(stdout);
    }

    // --- Tear down --------------------------------------------------------
    cross_node_barrier(ctxs, barrier_bufs);
    for (auto& c : ctxs) {
        if (c->comm) ncclCommDestroy(c->comm);
    }
    return 0;
}
