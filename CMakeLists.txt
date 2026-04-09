cmake_minimum_required(VERSION 3.22)
project(tensor_parallel_gemm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Default to A100 if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# Find CUDA toolkit (for cuBLAS)
find_package(CUDAToolkit REQUIRED)

# ── Kernel library (static) ──
add_library(gemm_kernels STATIC
    src/kernels/01_naive.cu
    src/kernels/02_coalesced.cu
    src/kernels/03_smem_tiling.cu
    src/kernels/04_1d_blocktile.cu
    src/kernels/05_2d_blocktile.cu
    src/kernels/06_vectorized.cu
    src/kernels/07_warptile.cu
    src/kernels/cublas_ref.cu
)
target_include_directories(gemm_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gemm_kernels PUBLIC CUDA::cublas)

# ── Single-GPU benchmark ──
add_executable(bench_single_gpu src/benchmark/bench_single_gpu.cu)
target_link_libraries(bench_single_gpu PRIVATE gemm_kernels)

# ── Multi-GPU targets (require NCCL) ──
find_library(NCCL_LIBRARY nccl
    HINTS ENV NCCL_HOME
    PATH_SUFFIXES lib lib64
)
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ENV NCCL_HOME
    PATH_SUFFIXES include
)

if(NCCL_LIBRARY AND NCCL_INCLUDE_DIR)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")

    # Tensor parallel library
    add_library(tensor_parallel STATIC
        src/tensor_parallel/tensor_parallel.cu
    )
    target_include_directories(tensor_parallel PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${NCCL_INCLUDE_DIR}
    )
    target_link_libraries(tensor_parallel PUBLIC gemm_kernels ${NCCL_LIBRARY})

    # Multi-GPU benchmark
    add_executable(bench_multi_gpu src/benchmark/bench_multi_gpu.cu)
    target_include_directories(bench_multi_gpu PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(bench_multi_gpu PRIVATE tensor_parallel)
else()
    message(WARNING "NCCL not found — multi-GPU targets disabled. "
                    "Set NCCL_HOME or install NCCL to enable.")
endif()
