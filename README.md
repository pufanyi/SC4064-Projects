# Scaling Matrix Multiplication: From CUDA Kernels to Multi-GPU Tensor Parallelism

**SC4064 GPU Programming — Course Project**
Nanyang Technological University

## Overview

This project explores GPU-accelerated General Matrix Multiplication (GEMM) across two dimensions:

1. **Single-GPU kernel optimization**: 7 progressively optimized CUDA kernels, each introducing a specific hardware-aware optimization, benchmarked against cuBLAS.
2. **Multi-GPU tensor parallelism**: Column-parallel and row-parallel linear layers using NCCL, with strong/weak scaling analysis and communication-compute overlap investigation.

The goal is to understand *why* each optimization works at the hardware level and *where* the bottleneck shifts from compute to communication as we scale.

---

## Why This Matters

GEMM is the computational core of deep learning — every linear layer, attention head, and convolution reduces to matrix multiplication. Understanding how to optimize it at the hardware level (memory coalescing, shared memory tiling, register blocking) and scale it across GPUs (tensor parallelism, NCCL collectives) is fundamental to building efficient training systems. This project bridges the gap between "call cuBLAS" and understanding *why* cuBLAS is fast.

---

## Project Structure

```
tensor-parallel-gemm/
├── Makefile                          # Build system
├── src/
│   ├── kernels/                      # Progressive GEMM optimizations
│   │   ├── kernels.cuh               # Kernel declarations
│   │   ├── 01_naive.cu               # Baseline: 1 thread = 1 output element
│   │   ├── 02_coalesced.cu           # Memory coalescing demonstration
│   │   ├── 03_smem_tiling.cu         # Shared memory tiling
│   │   ├── 04_1d_blocktile.cu        # Thread coarsening along M
│   │   ├── 05_2d_blocktile.cu        # 2D register blocking (TM×TN per thread)
│   │   ├── 06_vectorized.cu          # float4 vectorized loads + transposed smem
│   │   ├── 07_warptile.cu            # Warp-level tiling hierarchy
│   │   └── cublas_ref.cu             # cuBLAS reference wrapper
│   ├── tensor_parallel/
│   │   └── tensor_parallel.cu        # Column/Row parallel + Parallel MLP
│   ├── benchmark/
│   │   ├── bench_single_gpu.cu       # Correctness + GFLOPS benchmark
│   │   └── bench_multi_gpu.cu        # Scaling analysis benchmark
│   └── utils/
│       └── cuda_utils.cuh            # Error checking, timing, verification
├── scripts/
│   ├── run_benchmarks.sh             # Run everything
│   ├── nscc_job.sh                   # NSCC ASPIRE 2A job submission
│   └── plot_results.py               # Generate performance plots
├── results/                          # Benchmark outputs and plots
└── docs/
    └── optimization_notes.md         # Detailed optimization analysis
```

---

## Kernel Optimization Roadmap

Each kernel introduces one major optimization. The table below summarizes the progression:

| Kernel | Technique | Key Idea | Arithmetic Intensity |
|--------|-----------|----------|---------------------|
| 1. Naive | Baseline | 1 thread → 1 element, K global loads per FMA | O(1) FLOP/byte |
| 2. Coalesced | Memory coalescing | threadIdx.x → column for stride-1 access | O(1), fewer transactions |
| 3. Shared Memory | Tiling in SRAM | Load tile to smem, reuse TILE_SIZE times | O(TILE_SIZE) |
| 4. 1D Block Tile | Thread coarsening | Each thread computes TM=8 rows | O(TM × BK) |
| 5. 2D Block Tile | Register blocking | Each thread computes TM×TN=8×8 sub-tile | O(TM × TN) |
| 6. Vectorized | float4 loads | 128-bit transactions, transposed smem | Same, fewer instructions |
| 7. Warp Tile | Warp-level hierarchy | Block → Warp → Thread tiling | Same, better scheduling |

### Kernel 1: Naive
Each thread computes one element of C by iterating over K. No data reuse — every FMA requires two global memory loads. Entirely memory-bound, typically 1-2% of peak.

### Kernel 2: Global Memory Coalescing
Demonstrates the impact of thread-to-data mapping. Coalesced variant (threadIdx.x → column) allows the hardware to combine 32 individual 4-byte requests into one 128-byte transaction. We include an "uncoalesced" variant for direct comparison (5-10x difference).

### Kernel 3: Shared Memory Tiling
First major optimization. A 32×32 tile of A and B is loaded into shared memory (~20ns, ~19 TB/s vs ~400ns, ~2 TB/s for global). Each loaded element reused 32 times, reducing global traffic by 32x.

### Kernel 4: 1D Block Tiling
Each thread computes TM=8 rows. A single B shared memory load is reused across 8 accumulations. Block: (64, 8) = 512 threads.

### Kernel 5: 2D Block Tiling (Register Blocking)
The most important optimization. Each thread computes 8×8 sub-tile in registers. Outer product: each smem load pair feeds 64 FMAs. Block: BM=BN=128, BK=8, 256 threads.

### Kernel 6: Vectorized Memory Access
Uses `float4` (128-bit) loads and stores A transposed in smem (`As[k][m]`) to eliminate bank conflicts during compute.

### Kernel 7: Warp Tiling
Hierarchical tiling: Block (128×128) → Warp (32×64) → Thread (8×8). Threads within a warp access nearby smem, improving L1 hits and scheduling.

---

## Tensor Parallelism

### Column Parallelism
Weight W sharded column-wise. Each GPU: `Y_i = X @ W_i`. AllGather assembles full output. Used for first MLP layer.

### Row Parallelism
Weight W sharded row-wise, input X split correspondingly. Each GPU: `Y_i = X_i @ W_i`. AllReduce sums partials. Used for second MLP layer.

### Parallel MLP Block
Column-parallel (layer 1) → Row-parallel (layer 2) = only **one AllReduce** per block forward pass, matching Megatron-LM.

---

## Building & Running

### Prerequisites
- CUDA Toolkit ≥ 11.0, cuBLAS, NCCL, Python 3 + matplotlib

### Build
```bash
make all GPU_ARCH=sm_80    # A100
make all GPU_ARCH=sm_90    # H100
```

### Run
```bash
./build/bench_single_gpu                  # Single-GPU benchmark
./build/bench_multi_gpu 4                 # Multi-GPU (4 GPUs)
bash scripts/run_benchmarks.sh            # Full pipeline with plots
qsub scripts/nscc_job.sh                  # NSCC cluster
```

### Profiling
```bash
make profile                              # Nsight Compute
make timeline NUM_GPUS=4                  # Nsight Systems
```

---

## Evaluation Plan

1. **Kernel benchmarking**: GFLOPS at sizes 256–4096, identify memory→compute transition
2. **Roofline analysis**: Plot kernels on A100 roofline (19.5 TFLOPS, 2 TB/s)
3. **Strong scaling**: Fixed 4096×4096, 1→2→4 GPUs, measure parallel efficiency
4. **Weak scaling**: Fixed 2048×2048 per GPU, constant time is ideal
5. **Bottleneck crossover**: Communication time vs compute time as size grows

---

## References

1. Shoeybi et al. "Megatron-LM." arXiv:1909.08053, 2019.
2. Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
3. Williams et al. "Roofline Model." CACM, 2009.
4. Jia et al. "Dissecting NVIDIA Volta." arXiv:1804.06826, 2018.
5. [Simon Boehm — CUDA GEMM Optimization](https://siboehm.com/articles/22/CUDA-MMM)

## Team

- Aryan Jain (ARYAN017@e.ntu.edu.sg)
- Fanyi Pu (FPU001@e.ntu.edu.sg)
- Ze Hong Maxwell Au (MAU002@e.ntu.edu.sg)
