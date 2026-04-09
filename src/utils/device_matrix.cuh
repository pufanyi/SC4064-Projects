/*
 * DeviceMatrix
 * ============
 * Owns a device-side float matrix with dimensions, providing RAII memory
 * management and convenient initialization. Eliminates the pattern of
 * separate cudaMalloc + init_matrix + cudaMemcpy scattered across benchmarks.
 */

#pragma once

#include <vector>

#include "cuda_raii.cuh"
#include "cuda_utils.cuh"

class DeviceMatrix {
    CudaMemory<float> data_;
    int rows_ = 0;
    int cols_ = 0;

   public:
    DeviceMatrix() = default;

    DeviceMatrix(int rows, int cols) : data_(rows * cols), rows_(rows), cols_(cols) {}

    // Move operations (delegated to CudaMemory)
    DeviceMatrix(DeviceMatrix&&) = default;
    DeviceMatrix& operator=(DeviceMatrix&&) = default;
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    float* get() const { return data_.get(); }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int numel() const { return rows_ * cols_; }
    std::size_t bytes() const { return data_.bytes(); }

    void init_random(unsigned seed = 42) {
        std::vector<float> host(numel());
        init_matrix(host.data(), rows_, cols_, seed);
        data_.copy_from_host(host.data(), numel());
    }

    void zero() { data_.zero(); }

    void copy_from_host(const float* src) { data_.copy_from_host(src, numel()); }

    void copy_to_host(float* dst) const { data_.copy_to_host(dst, numel()); }
};
