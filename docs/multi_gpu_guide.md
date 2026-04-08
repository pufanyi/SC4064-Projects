# Multi-GPU Testing & Debugging Guide

This guide covers how to build, run, and debug the multi-GPU tensor parallel benchmark on a machine with 2+ GPUs and NCCL installed.

## Prerequisites

1. **2+ NVIDIA GPUs** (ideally same model, e.g., 2x A100 or 2x H100)
2. **NCCL library** installed (header `nccl.h` and `libnccl.so`)
3. **CUDA toolkit** 11.0+ (tested with 13.1)
4. **cuBLAS** (comes with CUDA toolkit)

### Installing NCCL

```bash
# Ubuntu/Debian (recommended)
sudo apt-get install libnccl2 libnccl-dev

# Or from NVIDIA's repo
# See: https://developer.nvidia.com/nccl/nccl-download

# Verify installation
ls /usr/include/nccl.h
ls /usr/lib/x86_64-linux-gnu/libnccl.so*
```

If NCCL is installed in a non-standard path, set `NCCL_HOME`:
```bash
export NCCL_HOME=/path/to/nccl
```

## Building

```bash
# Detect GPU architecture
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# A100 → sm_80, H100 → sm_90

# Build everything
GPU_ARCH=sm_80 make all

# Or build multi-GPU benchmark only
GPU_ARCH=sm_80 make bench_multi
```

If you hit linker errors for NCCL:
```bash
# Check NCCL_HOME points to the right place
NCCL_HOME=/usr make bench_multi
# Common locations: /usr, /usr/local, /opt/nccl
```

## Running the Multi-GPU Benchmark

```bash
# Basic: 2 GPUs, cuBLAS kernel
./build/bench_multi_gpu 2

# With specific kernel:
# kernel_id: 0=naive, 1=coalesced, 2=smem, 3=1d_blocktile,
#            4=2d_blocktile, 5=vectorized, 6=warptile, 7=cuBLAS
./build/bench_multi_gpu 2 7    # cuBLAS (default)
./build/bench_multi_gpu 2 6    # warptile kernel
./build/bench_multi_gpu 2 4    # 2d_blocktile kernel

# 4 GPUs with vectorized kernel
./build/bench_multi_gpu 4 5
```

### What Each Experiment Measures

| Experiment | What it tests | What to look for |
|---|---|---|
| **Exp 1: Strong Scaling** | Fixed work, more GPUs | GEMM time should ~halve per doubling |
| **Exp 2: Weak Scaling** | Fixed work/GPU | GEMM time should stay constant |
| **Exp 3: Comm/Compute Ratio** | Ratio vs matrix size | Ratio should decrease for larger matrices |
| **Exp 4: Ratio across Kernels** | All kernels at size=4096 | Faster kernels → higher ratio (comm becomes bottleneck) |
| **Exp 5: MLP Forward+Backward** | Full MLP block timing | Backward should be ~2-3x forward (more GEMMs) |
| **Exp 6: Comm-Compute Overlap** | Overlap vs no overlap | Speedup > 1.0x means overlap is effective |

## Expected Behavior & What to Debug

### Experiment 4 is the Key Result

This experiment directly addresses the proposal's question: *"how does the communication-to-computation ratio evolve as custom GEMM kernels are optimized?"*

Expected output pattern:
```
Kernel                GEMM(ms)    Comm(ms)     Ratio
1_naive                 50.000       0.500      0.01   ← comm negligible
2_coalesced             25.000       0.500      0.02
...
7_warptile               1.200       0.500      0.42   ← comm significant
cuBLAS                   0.800       0.500      0.63   ← comm dominant!
```

**Key insight**: As the local GEMM gets faster, communication becomes the bottleneck. This is the "crossover point" the proposal discusses.

### Experiment 6: Overlap Notes

The overlap experiment splits the M dimension into chunks and pipelines GEMM with AllReduce:

```
No overlap:   [--- GEMM ---][--- AllReduce ---]
With overlap: [GEMM chunk1][GEMM chunk2][GEMM chunk3][GEMM chunk4]
                           [AllRed 1   ][AllRed 2   ][AllRed 3   ][AllRed 4]
```

Overlap is most effective when:
- Comm and compute take roughly equal time (ratio ~ 1.0)
- Matrix is large enough to split into meaningful chunks
- For very small matrices or very fast kernels, overlap overhead may hurt

### Common Issues

**1. "NCCL error: unhandled system error"**
```bash
# Check GPU peer access
nvidia-smi topo -m
# Ensure GPUs can communicate (NVLink or PCIe)

# Try disabling P2P if it fails
export NCCL_P2P_DISABLE=1
./build/bench_multi_gpu 2
```

**2. "Requested N GPUs but only 1 available"**
```bash
# Verify GPUs are visible
nvidia-smi -L
# Check CUDA_VISIBLE_DEVICES is not restricting GPUs
echo $CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
```

**3. Backward pass produces wrong gradients**

To verify backward correctness, compare against a single-GPU reference:
```bash
# The backward pass does:
#   Column backward: dX = AllReduce(dY_i @ W_i^T), dW_i = X^T @ dY_i
#   Row backward:    dX_i = dY @ W_i^T, dW_i = X_i^T @ dY
#
# On 1 GPU, AllReduce is a no-op, so the result should match
# a single large GEMM. Use bench_multi_gpu with 1 GPU as baseline:
./build/bench_multi_gpu 1 7
```

**4. Custom kernels give wrong results in TP layer**

Custom kernels don't support the cuBLAS transpose trick natively. The backward pass uses `cublasSgeam` for explicit transposition before calling custom kernels. If results are wrong:
- Check that the transpose workspace is allocated correctly
- Verify the kernel handles non-square matrices (M != N != K)
- Some kernels (especially vectorized/warptile) require dimensions to be multiples of their block tile size (128). If `N/num_gpus` is not a multiple of 128, there may be edge-case issues.

**5. Overlap shows no speedup or slowdown**

- With only 2 GPUs, AllReduce is essentially a single send+recv, which is fast
- Overlap overhead (event recording, stream synchronization) may dominate for small data
- Try increasing `num_chunks` or matrix size

## NSCC Cluster Submission

Update `scripts/nscc_job.sh` for the new experiments:

```bash
#!/bin/bash
#PBS -l select=1:ngpus=4:ncpus=16:mem=64gb
#PBS -l walltime=01:00:00
#PBS -q gpu

cd $PBS_O_WORKDIR
module load cuda nccl

# Build
GPU_ARCH=sm_80 make clean all

# Run single-GPU benchmark
./build/bench_single_gpu > results/single_gpu_$(date +%Y%m%d).txt 2>&1

# Run multi-GPU with different kernels
for kernel_id in 0 1 2 3 4 5 6 7; do
    echo "=== Kernel $kernel_id ==="
    ./build/bench_multi_gpu 4 $kernel_id
done > results/multi_gpu_$(date +%Y%m%d).txt 2>&1

# Nsight Systems timeline
nsys profile -o results/timeline_mlp \
    ./build/bench_multi_gpu 4 7
```

## Architecture Summary

```
src/
├── kernels/
│   ├── kernels.cuh          # Kernel declarations
│   ├── gemm_dispatch.cuh    # NEW: Unified kernel selection (enum + dispatch)
│   ├── 01_naive.cu ... 07_warptile.cu
│   └── cublas_ref.cu
├── tensor_parallel/
│   └── tensor_parallel.cu   # UPDATED: +backward pass, +custom kernels, +overlap
├── benchmark/
│   ├── bench_single_gpu.cu  # Single-GPU correctness + GFLOPS
│   └── bench_multi_gpu.cu   # UPDATED: 6 experiments, kernel selection
└── utils/
    └── cuda_utils.cuh
```

### New Features Added

1. **`gemm_dispatch.cuh`**: Enum `GemmKernel` + `dispatch_gemm()` function to select any kernel
2. **Backward passes**: `column_parallel_backward()`, `row_parallel_backward()`, `parallel_mlp_backward()`
3. **Communication-compute overlap**: `row_parallel_forward_overlap()` using chunked pipelining
4. **Exp 4**: Comm/compute ratio across all kernels (directly answers proposal question)
5. **Exp 5**: MLP forward+backward timing
6. **Exp 6**: Overlap vs no-overlap comparison
