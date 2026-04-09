/*
 * CUDA RAII Wrappers
 * ==================
 * Move-only RAII classes for CUDA resources, eliminating manual
 * cudaMalloc/cudaFree, cudaStreamCreate/Destroy, etc.
 *
 * All classes follow the rule of five with deleted copy operations.
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

#include "cuda_utils.cuh"

// ---------------------------------------------------------------------------
// CudaMemory<T> — Owns a device allocation of `count` elements of type T
// ---------------------------------------------------------------------------
template <typename T = float>
class CudaMemory {
    T* ptr_ = nullptr;
    std::size_t count_ = 0;

   public:
    CudaMemory() = default;

    explicit CudaMemory(std::size_t count) : count_(count) {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~CudaMemory() {
        if (ptr_) cudaFree(ptr_);
    }

    // Move constructor
    CudaMemory(CudaMemory&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    // Move assignment
    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    // Non-copyable
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }

    void copy_from_host(const T* src, std::size_t n) {
        CUDA_CHECK(cudaMemcpy(ptr_, src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* dst, std::size_t n) const {
        CUDA_CHECK(cudaMemcpy(dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() { CUDA_CHECK(cudaMemset(ptr_, 0, bytes())); }
};

// ---------------------------------------------------------------------------
// CudaStream — Owns a cudaStream_t
// ---------------------------------------------------------------------------
class CudaStream {
    cudaStream_t stream_ = nullptr;

   public:
    CudaStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStream() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }
};

// ---------------------------------------------------------------------------
// CudaEvent — Owns a cudaEvent_t
// ---------------------------------------------------------------------------
class CudaEvent {
    cudaEvent_t event_ = nullptr;

   public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    ~CudaEvent() {
        if (event_) cudaEventDestroy(event_);
    }

    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) { other.event_ = nullptr; }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }

    void record(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(event_, stream)); }
    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

    float elapsed_ms(const CudaEvent& start) const {
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }
};

// ---------------------------------------------------------------------------
// CublasHandle — Owns a cublasHandle_t
// ---------------------------------------------------------------------------
class CublasHandle {
    cublasHandle_t handle_ = nullptr;

   public:
    CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
    ~CublasHandle() {
        if (handle_) cublasDestroy(handle_);
    }

    CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cublasDestroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }
    operator cublasHandle_t() const { return handle_; }
};
