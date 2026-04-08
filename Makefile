# ===========================================================================
# Tensor Parallel GEMM — Makefile
# ===========================================================================
# Targets:
#   bench_single   — Single-GPU kernel benchmark (correctness + GFLOPS)
#   bench_multi    — Multi-GPU tensor parallel benchmark (scaling analysis)
#   all            — Build everything
#   clean          — Remove build artifacts
#
# Environment:
#   CUDA_HOME      — CUDA toolkit root (default: /usr/local/cuda)
#   NCCL_HOME      — NCCL install root (default: /usr/local)
#   GPU_ARCH       — Target GPU architecture (default: sm_80 for A100)
# ===========================================================================

CUDA_HOME  ?= /usr/local/cuda
NCCL_HOME  ?= /usr
GPU_ARCH   ?= sm_80

NVCC       := $(CUDA_HOME)/bin/nvcc
NVCCFLAGS  := -std=c++17 -O3 -arch=$(GPU_ARCH) \
              --expt-relaxed-constexpr \
              -Xcompiler -Wall

# Include paths
INCLUDES   := -Isrc -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include

# Libraries
LDFLAGS    := -L$(CUDA_HOME)/lib64 -L$(NCCL_HOME)/lib
LIBS_SINGLE := -lcublas
LIBS_MULTI  := -lcublas -lnccl

# Source files
KERNEL_SRCS := src/kernels/01_naive.cu \
               src/kernels/02_coalesced.cu \
               src/kernels/03_smem_tiling.cu \
               src/kernels/04_1d_blocktile.cu \
               src/kernels/05_2d_blocktile.cu \
               src/kernels/06_vectorized.cu \
               src/kernels/07_warptile.cu \
               src/kernels/cublas_ref.cu

BUILD_DIR  := build

# ===========================================================================
# Targets
# ===========================================================================

.PHONY: all clean bench_single bench_multi

all: bench_single bench_multi

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

bench_single: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		src/benchmark/bench_single_gpu.cu $(KERNEL_SRCS) \
		$(LDFLAGS) $(LIBS_SINGLE) \
		-o $(BUILD_DIR)/bench_single_gpu

bench_multi: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		src/benchmark/bench_multi_gpu.cu \
		src/tensor_parallel/tensor_parallel.cu \
		$(KERNEL_SRCS) \
		$(LDFLAGS) $(LIBS_MULTI) \
		-o $(BUILD_DIR)/bench_multi_gpu

clean:
	rm -rf $(BUILD_DIR)

# Individual kernel test (useful during development)
test_kernel: $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		-DTEST_SINGLE_KERNEL \
		src/benchmark/bench_single_gpu.cu $(KERNEL_SRCS) \
		$(LDFLAGS) $(LIBS_SINGLE) \
		-o $(BUILD_DIR)/test_kernel

# Nsight Compute profiling target
profile: bench_single
	ncu --set full \
		--target-processes all \
		-o $(BUILD_DIR)/profile_report \
		$(BUILD_DIR)/bench_single_gpu

# Nsight Systems timeline
timeline: bench_multi
	nsys profile \
		-o $(BUILD_DIR)/timeline_report \
		$(BUILD_DIR)/bench_multi_gpu $(NUM_GPUS)
