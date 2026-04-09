/*
 * KernelRegistry — Singleton Kernel Manager
 * ==========================================
 * Each kernel .cu file self-registers via a static initializer.
 * Consumers iterate or look up kernels by index or name.
 */

#pragma once

#include <cstring>
#include <memory>
#include <vector>

#include "gemm_kernel.cuh"

class KernelRegistry {
   public:
    static KernelRegistry& instance() {
        static KernelRegistry reg;
        return reg;
    }

    /// Register a kernel. Called from static initializers in each .cu file.
    static int add(std::unique_ptr<GemmKernel> k) {
        auto& kernels = instance().kernels_;
        int id = static_cast<int>(kernels.size());
        kernels.push_back(std::move(k));
        return id;
    }

    /// Number of registered kernels.
    static int count() { return static_cast<int>(instance().kernels_.size()); }

    /// Get kernel by index.
    static const GemmKernel& get(int id) { return *instance().kernels_.at(id); }

    /// Get mutable kernel by index (for set_cublas_handle).
    static GemmKernel& get_mut(int id) { return *instance().kernels_.at(id); }

    /// Find kernel by name (linear scan, returns nullptr if not found).
    static const GemmKernel* find(const char* name) {
        for (auto& k : instance().kernels_) {
            if (std::strcmp(k->name(), name) == 0) return k.get();
        }
        return nullptr;
    }

    /// All registered kernels.
    static const std::vector<std::unique_ptr<GemmKernel>>& all() { return instance().kernels_; }

   private:
    KernelRegistry() = default;
    std::vector<std::unique_ptr<GemmKernel>> kernels_;
};
