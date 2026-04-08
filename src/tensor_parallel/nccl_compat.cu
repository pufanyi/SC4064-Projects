#include "../nccl.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct CallParams {
    const void *sendbuff = nullptr;
    void *recvbuff = nullptr;
    size_t count = 0;
    cudaStream_t stream = nullptr;
};

struct CommGroupState {
    explicit CommGroupState(std::vector<int> devices_)
        : devices(std::move(devices_)),
          pending(devices.size()),
          scratch(devices.size(), nullptr),
          scratch_capacity(devices.size(), 0) {}

    std::vector<int> devices;
    std::vector<CallParams> pending;
    std::vector<float *> scratch;
    std::vector<size_t> scratch_capacity;
    int refs = 0;
    int arrived = 0;
    int generation = 0;
    ncclResult_t last_status = ncclSuccess;
    std::mutex mutex;
    std::condition_variable cv;
};

std::mutex registry_mutex;
std::unordered_map<int, std::shared_ptr<CommGroupState>> registry;
std::atomic<int> next_group_id{1};

const char *cuda_error_to_string(cudaError_t err) {
    return cudaGetErrorString(err);
}

ncclResult_t check_cuda(cudaError_t err) {
    if (err == cudaSuccess) return ncclSuccess;
    if (err == cudaErrorPeerAccessAlreadyEnabled) return ncclSuccess;
    fprintf(stderr, "nccl_compat CUDA error: %s\n", cuda_error_to_string(err));
    return ncclUnhandledCudaError;
}

std::shared_ptr<CommGroupState> get_group(int group_id) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    return registry.at(group_id);
}

__global__ void reduce_sum_kernel(const float *scratch,
                                  float *out,
                                  size_t count,
                                  int nranks) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float sum = 0.0f;
    for (int r = 0; r < nranks; r++) {
        sum += scratch[r * count + idx];
    }
    out[idx] = sum;
}

ncclResult_t enqueue_allgather(CommGroupState &group) {
    const int nranks = static_cast<int>(group.devices.size());
    const size_t bytes = group.pending[0].count * sizeof(float);

    for (int dst = 0; dst < nranks; dst++) {
        cudaError_t err = cudaSetDevice(group.devices[dst]);
        if (err != cudaSuccess) return check_cuda(err);

        char *dst_base = static_cast<char *>(group.pending[dst].recvbuff);
        cudaStream_t stream = group.pending[dst].stream;

        for (int src = 0; src < nranks; src++) {
            char *dst_ptr = dst_base + src * bytes;
            const void *src_ptr = group.pending[src].sendbuff;
            if (group.devices[src] == group.devices[dst]) {
                err = cudaMemcpyAsync(dst_ptr, src_ptr, bytes,
                                      cudaMemcpyDeviceToDevice, stream);
            } else {
                err = cudaMemcpyPeerAsync(dst_ptr, group.devices[dst],
                                          src_ptr, group.devices[src], bytes,
                                          stream);
            }
            if (err != cudaSuccess) return check_cuda(err);
        }
    }

    return ncclSuccess;
}

ncclResult_t ensure_scratch(CommGroupState &group, int rank, size_t count) {
    if (group.scratch_capacity[rank] >= count) return ncclSuccess;

    cudaError_t err = cudaSetDevice(group.devices[rank]);
    if (err != cudaSuccess) return check_cuda(err);

    if (group.scratch[rank]) {
        err = cudaFree(group.scratch[rank]);
        if (err != cudaSuccess) return check_cuda(err);
    }

    err = cudaMalloc(&group.scratch[rank], count * sizeof(float));
    if (err != cudaSuccess) return check_cuda(err);

    group.scratch_capacity[rank] = count;
    return ncclSuccess;
}

ncclResult_t enqueue_allreduce(CommGroupState &group) {
    const int nranks = static_cast<int>(group.devices.size());
    const size_t count = group.pending[0].count;
    const size_t bytes = count * sizeof(float);

    for (int dst = 0; dst < nranks; dst++) {
        ncclResult_t status = ensure_scratch(group, dst, count * nranks);
        if (status != ncclSuccess) return status;

        cudaError_t err = cudaSetDevice(group.devices[dst]);
        if (err != cudaSuccess) return check_cuda(err);

        cudaStream_t stream = group.pending[dst].stream;
        char *scratch_base = reinterpret_cast<char *>(group.scratch[dst]);
        for (int src = 0; src < nranks; src++) {
            char *dst_ptr = scratch_base + src * bytes;
            const void *src_ptr = group.pending[src].sendbuff;
            if (group.devices[src] == group.devices[dst]) {
                err = cudaMemcpyAsync(dst_ptr, src_ptr, bytes,
                                      cudaMemcpyDeviceToDevice, stream);
            } else {
                err = cudaMemcpyPeerAsync(dst_ptr, group.devices[dst],
                                          src_ptr, group.devices[src], bytes,
                                          stream);
            }
            if (err != cudaSuccess) return check_cuda(err);
        }

        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        reduce_sum_kernel<<<blocks, threads, 0, stream>>>(
            group.scratch[dst],
            static_cast<float *>(group.pending[dst].recvbuff),
            count, nranks);
        err = cudaGetLastError();
        if (err != cudaSuccess) return check_cuda(err);
    }

    return ncclSuccess;
}

template <typename LaunchFn>
ncclResult_t rendezvous(ncclComm_t comm, const CallParams &params, LaunchFn launch) {
    auto group = get_group(comm->group_id);
    std::unique_lock<std::mutex> lock(group->mutex);
    const int generation = group->generation;

    group->pending[comm->rank] = params;
    group->arrived++;

    if (group->arrived == comm->nranks) {
        lock.unlock();
        ncclResult_t status = launch(*group);
        lock.lock();
        group->last_status = status;
        group->arrived = 0;
        group->generation++;
        group->cv.notify_all();
    } else {
        group->cv.wait(lock, [&] { return group->generation != generation; });
    }

    cudaError_t err = cudaSetDevice(group->devices[comm->rank]);
    if (err != cudaSuccess) return check_cuda(err);

    return group->last_status;
}

}  // namespace

extern "C" {

const char *ncclGetErrorString(ncclResult_t result) {
    switch (result) {
        case ncclSuccess: return "ncclSuccess";
        case ncclInvalidArgument: return "ncclInvalidArgument";
        case ncclUnhandledCudaError: return "ncclUnhandledCudaError";
        case ncclSystemError: return "ncclSystemError";
        default: return "ncclUnknownError";
    }
}

ncclResult_t ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist) {
    if (!comms || !devlist || ndev <= 0) return ncclInvalidArgument;

    std::vector<int> devices(devlist, devlist + ndev);
    auto group = std::make_shared<CommGroupState>(devices);
    group->refs = ndev;

    for (int i = 0; i < ndev; i++) {
        for (int j = 0; j < ndev; j++) {
            if (i == j) continue;
            int can_access = 0;
            cudaError_t err = cudaDeviceCanAccessPeer(
                &can_access, devices[i], devices[j]);
            if (err != cudaSuccess) return check_cuda(err);
            if (!can_access) continue;

            err = cudaSetDevice(devices[i]);
            if (err != cudaSuccess) return check_cuda(err);
            err = cudaDeviceEnablePeerAccess(devices[j], 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                return check_cuda(err);
            }
        }
    }

    const int group_id = next_group_id.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        registry[group_id] = group;
    }

    for (int rank = 0; rank < ndev; rank++) {
        comms[rank] = new ncclCommStub{ndev, rank, group_id};
    }

    return ncclSuccess;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    if (!comm) return ncclInvalidArgument;

    std::shared_ptr<CommGroupState> group;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = registry.find(comm->group_id);
        if (it == registry.end()) {
            delete comm;
            return ncclSuccess;
        }
        group = it->second;
        group->refs--;
        if (group->refs == 0) {
            registry.erase(it);
        }
    }

    if (group && group->refs == 0) {
        for (size_t rank = 0; rank < group->devices.size(); rank++) {
            if (!group->scratch[rank]) continue;
            cudaError_t err = cudaSetDevice(group->devices[rank]);
            if (err == cudaSuccess) cudaFree(group->scratch[rank]);
        }
    }

    delete comm;
    return ncclSuccess;
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount,
                           ncclDataType_t datatype, ncclComm_t comm,
                           cudaStream_t stream) {
    if (!comm || datatype != ncclFloat) return ncclInvalidArgument;
    return rendezvous(comm, CallParams{sendbuff, recvbuff, sendcount, stream},
                      [](CommGroupState &group) { return enqueue_allgather(group); });
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
    if (!comm || datatype != ncclFloat || op != ncclSum) return ncclInvalidArgument;
    return rendezvous(comm, CallParams{sendbuff, recvbuff, count, stream},
                      [](CommGroupState &group) { return enqueue_allreduce(group); });
}

ncclResult_t ncclGroupStart(void) {
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void) {
    return ncclSuccess;
}

}  // extern "C"
