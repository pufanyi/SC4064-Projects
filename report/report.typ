// ═══════════════════════════════════════════════════════════════════════
// Typst Report — Scaling Matrix Multiplication
// ═══════════════════════════════════════════════════════════════════════

#set document(
  title: "Scaling Matrix Multiplication: From CUDA Kernels to Multi-GPU Tensor Parallelism",
  author: ("Aryan Jain", "Fanyi Pu", "Ze Hong Maxwell Au"),
)

#set page(
  paper: "a4",
  margin: (x: 2.2cm, y: 2.4cm),
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(8pt, fill: gray)
      Scaling Matrix Multiplication #h(1fr) SC4064 GPU Programming
    ]
  },
)

#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.4em, below: 0.8em)
#set math.equation(numbering: "(1)")
#show figure: set block(above: 1.2em, below: 1.2em)
#show figure.caption: set text(size: 9pt)

// ── Helper ───────────────────────────────────────────────────────────
#let fig(path, caption, width: 95%) = {
  figure(
    image(path, width: width),
    caption: caption,
  )
}

// ── Title ────────────────────────────────────────────────────────────
#align(center)[
  #block(above: 2em, below: 0.6em)[
    #text(16pt, weight: "bold")[Scaling Matrix Multiplication:\ From CUDA Kernels to Multi-GPU Tensor Parallelism]
  ]
  #text(11pt)[
    Aryan Jain#super[\*] #h(1.5em) Fanyi Pu#super[\*] #h(1.5em) Ze Hong Maxwell Au#super[\*]
  ] \
  #text(9pt, fill: gray)[
    School of Computer Science and Engineering, Nanyang Technological University \
    #super[\*]Equal contribution. Authors listed in alphabetical order.
  ]
  #v(0.5em)
  #text(9pt)[SC4064 GPU Programming --- Course Project Report]
]

#v(0.8em)

// ═════════════════════════════════════════════════════════════════════
// Abstract
// ═════════════════════════════════════════════════════════════════════
#block(inset: (x: 2em))[
  #text(weight: "bold")[Abstract.]
  General Matrix Multiplication (GEMM) is the computational backbone of modern deep learning.
  This report presents a systematic study across two dimensions: (i) progressive single-GPU CUDA kernel optimization through seven stages---from naive global memory access to warp-level tiling---benchmarked against cuBLAS on NVIDIA H100 GPUs; and (ii) multi-GPU tensor parallelism using NCCL, implementing column-parallel and row-parallel linear layers following the Megatron-LM paradigm.
  Our best custom kernel achieves 63% of cuBLAS throughput at 32.8 TFLOPS.
  Scaling experiments on 8$times$H100 GPUs with matrices up to $32768 times 32768$ demonstrate $7.6 times$ strong-scaling speedup and 400 TFLOPS aggregate throughput. As local GEMM kernels approach peak performance, the communication-to-computation ratio rises from 0.20 (naive kernel) to 0.85 (cuBLAS), quantifying the crossover where inter-GPU communication becomes the dominant bottleneck.
]

#v(0.6em)

// ═════════════════════════════════════════════════════════════════════
= Introduction
// ═════════════════════════════════════════════════════════════════════

GEMM operations underpin virtually all compute-intensive workloads in deep learning @vaswani2017attention @jia2018dissecting. In Transformer architectures, multi-head attention and feed-forward layers are fundamentally matrix multiplications. As models scale beyond the memory capacity of a single accelerator, _tensor parallelism_ @shoeybi2019megatron has become indispensable, distributing weight matrices across GPUs at the cost of inter-GPU communication.

While vendor-tuned libraries such as cuBLAS @nvidia_cublas_2026 deliver near-optimal single-GPU performance, they abstract away the complex interaction between hardware-level compute intensity and system-level communication latency. This project bridges that gap by:
+ Implementing seven progressively optimized CUDA GEMM kernels to understand hardware-level constraints (memory coalescing, shared memory tiling, register blocking, warp scheduling).
+ Building a distributed tensor-parallel linear layer (forward and backward) using NCCL, including a complete parallel MLP block.
+ Quantifying how local kernel efficiency impacts multi-GPU scalability, identifying the _crossover point_ where communication dominates.

All experiments are conducted on the hardware described in @sec:setup.

== Experimental Setup <sec:setup>

@tab:setup summarizes the hardware and software environment used for all experiments.

#figure(
  table(
    columns: (auto, auto),
    inset: 6pt,
    align: (left, left),
    stroke: none,
    table.hline(),
    table.header([*Component*], [*Specification*]),
    table.hline(stroke: 0.5pt),
    [GPU], [8 $times$ NVIDIA H100 80GB HBM3 (SXM5)],
    [GPU Interconnect], [NVLink 4.0, all-to-all NV18 (900 GB/s bidirectional per GPU)],
    [GPU Compute], [132 SMs, 1980 MHz boost, sm\_90, 33.5 TFLOPS FP32 peak],
    [GPU Memory], [80 GB HBM3, $tilde$3.35 TB/s bandwidth per GPU],
    [CPU], [2 $times$ Intel Xeon Gold 6448Y (32 cores / 64 threads each, 128 threads total)],
    [System Memory], [2 TB DDR5],
    [OS], [Ubuntu 24.04.4 LTS (kernel 5.14.0)],
    [CUDA Toolkit], [13.1],
    [NCCL], [2.29.3],
    [GPU Driver], [550.90.07],
    table.hline(),
  ),
  caption: [Hardware and software configuration.],
) <tab:setup>

The 8 GPUs are fully connected via NVLink 4.0 with NV18 topology (18 NVLink connections per GPU pair), distributed across 2 NUMA nodes (GPUs 0--3 on NUMA 0, GPUs 4--7 on NUMA 1). This provides a uniform high-bandwidth, low-latency interconnect for NCCL collectives, eliminating PCIe bottlenecks.

// ═════════════════════════════════════════════════════════════════════
= Background
// ═════════════════════════════════════════════════════════════════════

== GPU Memory Hierarchy

The performance of GEMM kernels is governed by the GPU's memory hierarchy. @tab:memhier summarizes the key levels for the H100.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, right, right, right, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Level*], [*Size*], [*Bandwidth*], [*Latency*], [*Scope*],
    ),
    table.hline(stroke: 0.5pt),
    [Registers], [256 KB/SM], [$tilde 20$ TB/s], [$0$ cycles], [Per-thread],
    [Shared Memory], [228 KB/SM], [$tilde 19$ TB/s], [$tilde 20$ ns], [Per-block],
    [L2 Cache], [50 MB], [$tilde 5$ TB/s], [$tilde 200$ ns], [Global],
    [HBM3 (Global)], [80 GB], [$tilde 3.35$ TB/s], [$tilde 400$ ns], [Global],
    table.hline(),
  ),
  caption: [H100 GPU memory hierarchy. Each optimization moves data access to a faster level.],
) <tab:memhier>

Every kernel optimization in this work moves the _effective_ data access pattern from a slower to a faster level, or reduces total memory transactions.

== Roofline Model

The _roofline model_ @williams2009roofline characterizes achievable performance as a function of _operational intensity_ (OI, FLOP/byte):

$ P_"max" = min(pi, beta times "OI") $

where $pi$ is peak compute throughput (33.5 TFLOPS FP32 for H100) and $beta$ is peak memory bandwidth (3.35 TB/s). The _ridge point_ at $"OI" = pi / beta approx 10$ FLOP/byte separates memory-bound (left) from compute-bound (right) regimes. For GEMM of size $N$, the theoretical OI is $2N^3 slash (12N^2) = N slash 6$, which exceeds the ridge point for $N >= 60$---in practice, only optimized kernels achieve this.

== Tensor Parallelism

Tensor parallelism @shoeybi2019megatron distributes the weight matrices of a linear layer $Y = X W$ across $p$ GPUs. Two complementary partitioning strategies exist.

=== Column Parallelism

The weight matrix $W in RR^(K times N)$ is partitioned column-wise into $p$ shards:
$ W = [W_1 | W_2 | dots.c | W_p], quad W_i in RR^(K times N\/p) $
The input $X in RR^(M times K)$ is replicated on all GPUs. Each GPU $i$ computes a local GEMM:
$ Y_i = X W_i in RR^(M times N\/p) $
The full output is assembled via an `AllGather` collective:
$ Y = [Y_1 | Y_2 | dots.c | Y_p] = X W in RR^(M times N) $

*Backward pass.* Given upstream gradient $dif Y = [dif Y_1 | dots.c | dif Y_p]$, each GPU $i$ computes:
$ dif W_i = X^top dif Y_i in RR^(K times N\/p) quad "(local, no communication)" $
$ dif X_i = dif Y_i W_i^top in RR^(M times K) quad "(partial input gradient)" $
Since each GPU sees only its column shard, $dif X_i$ captures only the contribution from $W_i$. The full input gradient requires an `AllReduce` (sum):
$ dif X = sum_(i=1)^p dif X_i = dif Y W^top $

=== Row Parallelism

The weight matrix is partitioned row-wise:
$ W = mat(W_1; W_2; dots.v; W_p), quad W_i in RR^(K\/p times N) $
The input $X in RR^(M times K)$ is correspondingly split column-wise: $X = [X_1 | X_2 | dots.c | X_p]$ with $X_i in RR^(M times K\/p)$. Each GPU $i$ computes:
$ Y_i = X_i W_i in RR^(M times N) quad "(partial sum)" $
The full output is obtained by an `AllReduce`:
$ Y = sum_(i=1)^p Y_i = sum_(i=1)^p X_i W_i = X W $

*Backward pass.* Given upstream gradient $dif Y in RR^(M times N)$ (replicated after the forward `AllReduce`):
$ dif W_i = X_i^top dif Y in RR^(K\/p times N) quad "(local)" $
$ dif X_i = dif Y W_i^top in RR^(M times K\/p) quad "(local)" $
No communication is needed in the backward pass---$dif X_i$ is precisely the gradient for the local input shard.

=== Composing a Parallel MLP Block

A standard Transformer MLP block computes $"MLP"(X) = sigma(X W_1) W_2$, where $sigma$ is an activation function. Following Megatron-LM, we compose column-parallel (first layer) with row-parallel (second layer):

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Step*], [*Operation*], [*Per-GPU Compute*], [*Communication*]),
    table.hline(stroke: 0.5pt),
    [Fwd Layer 1], [Column parallel], [$H_i = X W_(1,i)$, #h(0.3em) $H_i in RR^(M times H\/p)$], [None],
    [Fwd Activation], [Element-wise], [$H_i <- sigma(H_i)$], [None],
    [Fwd Layer 2], [Row parallel], [$Y_i = H_i W_(2,i)$, #h(0.3em) $Y_i in RR^(M times N)$], [`AllReduce`($Y$)],
    [Bwd Layer 2], [Row backward], [$dif W_(2,i), dif H_i$ (local)], [None],
    [Bwd Layer 1], [Column backward], [$dif W_(1,i)$ (local), $dif X_i$ (partial)], [`AllReduce`($dif X$)],
    table.hline(),
  ),
  caption: [Communication pattern in a parallel MLP block. Only two `AllReduce` operations are needed: one in the forward pass and one in the backward pass.],
) <tab:mlp_comm>

The key insight (@tab:mlp_comm) is that column-parallel's output $H_i$ is _already_ the correct input shard for the subsequent row-parallel layer---no `AllGather` is needed between layers. This reduces per-block communication to exactly *one `AllReduce` forward* and *one `AllReduce` backward*.

=== Communication Cost Model

For a ring-based `AllReduce` with $p$ GPUs, message size $S$ bytes, per-hop bandwidth $B$, and latency $L$:
$ T_"AllReduce" approx 2(p-1)/p dot S/B + 2(p-1) dot L $
The first term (bandwidth) dominates for large messages. For column-parallel `AllGather`:
$ T_"AllGather" approx (p-1)/p dot S/B + (p-1) dot L $
As $p arrow infinity$, both approach $S\/B$ (bandwidth-limited). This means communication time depends primarily on _data volume_ ($S = M N dot 4$ bytes for FP32) and interconnect bandwidth, not on the local GEMM kernel speed---a fact we verify experimentally in @sec:ratio.

// ═════════════════════════════════════════════════════════════════════
= Single-GPU Kernel Optimization
// ═════════════════════════════════════════════════════════════════════

== Optimization Roadmap

We implement seven CUDA kernels, each introducing one major optimization. @tab:kernels summarizes the progression.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    table.header(
      [*Kernel*], [*Technique*], [*Key Idea*], [*Effect*],
    ),
    table.hline(stroke: 0.5pt),
    [1. Naive], [Baseline], [1 thread $arrow.r$ 1 element, $K$ global loads/FMA], [$cal(O)(1)$ FLOP/byte],
    [2. Coalesced], [Memory coalescing], [threadIdx.x $arrow.r$ column for stride-1], [Fewer transactions],
    [3. Shared Mem], [SRAM tiling], [$32 times 32$ tiles reused $32 times$], [$32 times$ less traffic],
    [4. 1D BlockTile], [Thread coarsening], [Each thread $arrow.r$ TM$=$8 rows], [Better smem reuse],
    [5. 2D BlockTile], [Register blocking], [TM$times$TN$= 8 times 8$ per thread], [$8 times$ fewer smem reads],
    [6. Vectorized], [`float4` loads], [128-bit transactions + transposed smem], [Fewer instructions],
    [7. WarpTile], [Warp-level tiling], [Block $arrow.r$ Warp $arrow.r$ Thread hierarchy], [Better L1 locality],
    table.hline(),
  ),
  caption: [Progressive GEMM kernel optimization roadmap.],
) <tab:kernels>

All kernels pass correctness verification against a CPU reference at $M = N = K = 256$ (max absolute error $< 10^(-5)$).

== Results and Analysis

#fig(
  "../results/figures/kernel_gflops.pdf",
  [GFLOPS comparison across all kernel optimization stages and matrix sizes. cuBLAS serves as the reference. Performance of optimized kernels (2D BlockTile, Vectorized, WarpTile) grows significantly with matrix size as compute becomes the bottleneck.],
) <fig:gflops>

@fig:gflops shows the performance progression. Key observations:

- *Naive and Coalesced kernels* plateau at $tilde 6{,}300$ GFLOPS regardless of matrix size, confirming they are memory-bandwidth-bound. The coalesced variant shows a $12.7 times$ speedup over the deliberately uncoalesced version (498 vs. 6,326 GFLOPS at $N = 4096$), demonstrating the critical importance of memory access patterns.

- *Shared memory tiling* (Kernel 3) improves throughput to $tilde 9{,}000$ GFLOPS by reducing global memory traffic by $32 times$, but remains below the compute-bound regime.

- *Register blocking* (Kernels 4--7) produces the most dramatic gains. The 2D block tile reaches 22,044 GFLOPS at $N = 4096$---a $3.5 times$ improvement over shared memory alone. Vectorized loads push this to 32,612 GFLOPS at $N = 4096$, sustaining $tilde 32{,}000$ GFLOPS through $N = 16384$.

- *cuBLAS* achieves 53,304 GFLOPS at $N = 16384$ ($tilde 162%$ of our best), leveraging Tensor Core paths and auto-tuning unavailable to our FP32-only kernels.

#fig(
  "../results/figures/cublas_percentage.pdf",
  [Each kernel's throughput as a percentage of cuBLAS. The Vectorized kernel reaches 63% of cuBLAS at $N = 4096$. The gap narrows significantly from Kernel 1 (12%) to Kernel 6 (63%).],
  width: 75%,
) <fig:pct>

@fig:pct shows the relative performance trajectory. The gap between custom kernels and cuBLAS decreases monotonically with optimization level, from 12% (naive) to 63% (vectorized). The remaining gap is primarily due to cuBLAS's use of Tensor Cores for FP32 accumulation and extensive auto-tuning.

#fig(
  "../results/figures/roofline.pdf",
  [Roofline analysis at $N = 4096$. Kernels progress from left (low effective OI, memory-bound) to right (high OI, approaching compute-bound). The ridge point at 10 FLOP/byte separates the two regimes. cuBLAS crosses into the compute-bound region.],
  width: 75%,
) <fig:roofline>

The roofline analysis (@fig:roofline) provides a unified view. Naive/Coalesced kernels sit in the memory-bound region with effective OI $< 2$ FLOP/byte. Each optimization stage moves kernels rightward: shared memory tiling ($tilde 2.7$), block tiling ($tilde 5$--$6.6$), vectorized loads ($tilde 9.7$). cuBLAS crosses the ridge point ($tilde 15.4$ FLOP/byte), entering the compute-bound regime.

// ═════════════════════════════════════════════════════════════════════
= Multi-GPU Tensor Parallelism
// ═════════════════════════════════════════════════════════════════════

We implement the tensor-parallel primitives described in @tab:mlp_comm using NCCL. Each GPU runs in a separate host thread with its own CUDA stream and cuBLAS handle; NCCL communicators are initialized via `ncclCommInitAll`. The implementation includes:

- *Column-parallel forward/backward*: local GEMM + `ncclAllGather` (forward) / `ncclAllReduce` (backward) as derived in Section 2.3.1--2.3.2.
- *Row-parallel forward/backward*: local GEMM + `ncclAllReduce` (forward) / no communication (backward).
- *Parallel MLP block*: composed column $arrow.r$ row parallelism with forward and backward passes, matching the two-`AllReduce` pattern in @tab:mlp_comm.
- *Communication-compute overlap*: the output matrix is chunked along the $M$ dimension; each chunk's `AllReduce` is pipelined with the next chunk's GEMM on a separate CUDA stream, synchronized via `cudaEvent`.

All multi-GPU experiments use 8$times$H100 GPUs connected via NVLink. For backward passes involving custom (non-cuBLAS) kernels, explicit transposition is performed using `cublasSgeam` since our custom kernels operate on row-major data without built-in transpose support.

== Strong Scaling <sec:strong>

#fig(
  "../results/figures/strong_scaling.pdf",
  [Strong scaling: total wall-clock time vs. number of GPUs for five matrix sizes up to $32768$. Dashed lines show ideal linear speedup. Larger matrices scale significantly better.],
  width: 72%,
) <fig:strong>

@fig:strong shows that the $N = 32768$ workload scales from 1331 ms (1 GPU) to 176 ms (8 GPUs)---a $7.6 times$ speedup, or *94.5% parallel efficiency*, achieving 400 TFLOPS aggregate throughput. Even $N = 8192$ achieves a respectable $4.1 times$ speedup. The $N = 2048$ case sees diminishing returns as communication overhead dominates at small matrix sizes.

#fig(
  "../results/figures/strong_scaling_efficiency.pdf",
  [Parallel efficiency $eta = T_1 / (p dot T_p)$ for strong scaling. At $N = 32768$, efficiency remains above 94% at 8 GPUs. Super-linear efficiency at small GPU counts is due to improved cache utilization.],
  width: 72%,
) <fig:efficiency>

@fig:efficiency shows that large matrices maintain near-ideal efficiency. At $N = 32768$, efficiency stays above 94% even at 8 GPUs because communication ($tilde 11$ ms) is dwarfed by compute ($tilde 166$ ms). Super-linear speedup at $p = 2$ for $N = 8192$ occurs because the halved per-GPU working set fits better in L2 cache.

== Weak Scaling

#fig(
  "../results/figures/weak_scaling.pdf",
  [Weak scaling with fixed $2048 times 2048$ local workload per GPU. Left: total time increases modestly from 0.44 ms to 1.12 ms. Right: aggregate throughput scales from 39.5 to 123.2 TFLOPS ($3.1 times$ at $8 times$ GPUs).],
) <fig:weak>

Under weak scaling (@fig:weak), total time increases from 0.44 ms (1 GPU) to 1.12 ms (8 GPUs)---a $2.6 times$ overhead for $8 times$ the total work. Aggregate throughput reaches 123.2 TFLOPS, which is $3.1 times$ the single-GPU baseline. The sub-linear throughput scaling is due to communication overhead growing with the number of participants in the `AllGather` collective.

== Communication-Compute Analysis <sec:ratio>

This is the central experiment of the project, directly addressing the question: _how does the communication-to-computation ratio evolve as custom GEMM kernels are optimized?_

#fig(
  "../results/figures/comm_compute_ratio_size.pdf",
  [GEMM time vs. communication time at different matrix sizes (8 GPUs, cuBLAS kernel). The ratio decreases from 0.99 at $N = 2048$ to 0.07 at $N = 32768$ as compute grows $O(N^3)$ while communication grows $O(N^2)$.],
  width: 72%,
) <fig:ratio_size>

@fig:ratio_size demonstrates the cubic-vs-quadratic scaling law. At $N = 2048$, communication (0.47 ms) nearly equals compute (0.48 ms), yielding a ratio of 0.99. At $N = 32768$, compute grows to 166 ms while communication is only 11 ms---a ratio of *0.07*. This confirms that tensor parallelism is most effective for the large matrix sizes encountered in modern Transformer models (e.g., hidden dimension 12288--16384 in GPT-3/LLaMA).

#fig(
  "../results/figures/comm_compute_ratio_kernel.pdf",
  [Left: absolute GEMM and communication time per kernel at $N = 4096$ on 8 GPUs. Communication time is constant ($tilde 0.64$ ms) across kernels. Right: the communication-to-compute ratio rises monotonically from 0.20 (naive) to 0.85 (cuBLAS) as kernels get faster.],
) <fig:ratio_kernel>

@fig:ratio_kernel is the *key result*. Communication time is constant across kernels ($tilde 0.64$ ms)---it depends only on data volume and interconnect bandwidth. As local GEMM time decreases from 3.15 ms (naive) to 0.75 ms (cuBLAS), the ratio rises from 0.20 to 0.85. This means:

- For *naive kernels*, communication is negligible (20% of compute)---further kernel optimization yields direct end-to-end speedup.
- For *cuBLAS*, communication consumes 85% of compute time---the system is approaching the _communication-bound_ regime where kernel optimization alone provides diminishing returns.

This crossover is the fundamental tension in distributed deep learning: local compute efficiency and system-level communication efficiency must be co-optimized.

== MLP Forward and Backward

#fig(
  "../results/figures/mlp_fwd_bwd.pdf",
  [Parallel MLP block timing on 8 GPUs. Backward pass takes 1.1$times$--1.7$times$ the forward pass, consistent with the additional GEMM operations for weight and input gradients.],
  width: 62%,
) <fig:mlp>

@fig:mlp shows the MLP block timing. The backward-to-forward ratio increases with matrix size (1.1$times$ at $N = 2048$ to 1.7$times$ at $N = 8192$), consistent with the backward pass requiring three GEMM operations per layer versus one in the forward pass, with the additional operations becoming more dominant as the compute-to-overhead ratio increases.

== Communication-Compute Overlap

#fig(
  "../results/figures/overlap_comparison.pdf",
  [Row-parallel forward with and without communication-compute overlap (4 chunks). Overlap provides marginal speedup ($1.02 times$) only at $N = 8192$ and introduces overhead at smaller sizes.],
  width: 62%,
) <fig:overlap>

The overlap experiment (@fig:overlap) splits the output along the M dimension into 4 chunks, pipelining each chunk's `AllReduce` with the next chunk's GEMM on separate CUDA streams. Results show:

- At $N = 2048$ and $4096$, overlap *hurts* performance (0.93$times$ and 0.96$times$) due to event synchronization overhead and reduced per-chunk GEMM efficiency.
- At $N = 8192$, a marginal $1.02 times$ speedup appears as chunk sizes become large enough to amortize overhead.

On H100 with NVLink, `AllReduce` latency is already very low ($tilde 1$ ms for 256 MB), limiting overlap opportunity. This technique would likely be more beneficial on systems with higher communication latency (e.g., PCIe or cross-node InfiniBand).

// ═════════════════════════════════════════════════════════════════════
= Conclusion
// ═════════════════════════════════════════════════════════════════════

This project provides a comprehensive study of GEMM optimization and tensor parallelism across two scales:

*Single-GPU.* Seven progressively optimized CUDA kernels demonstrate the transition from memory-bound ($tilde 6{,}300$ GFLOPS) to compute-bound ($tilde 32{,}600$ GFLOPS) operation. The most impactful optimization is 2D register blocking (Kernel 5), which reduces shared memory reads per FMA from 2 to 0.25. Our best kernel achieves 63% of cuBLAS throughput on FP32---the remaining gap is attributable to cuBLAS's Tensor Core utilization and auto-tuning.

*Multi-GPU.* Tensor parallelism on 8$times$H100 achieves $5.6 times$ strong-scaling speedup for $N = 8192$ and 290 TFLOPS aggregate throughput. The central finding is the communication-compute crossover: as local kernels approach peak efficiency, the communication-to-computation ratio rises from 0.20 to 0.84, quantifying the diminishing returns of kernel optimization alone in distributed settings.

These results underscore that efficient distributed deep learning requires *co-optimization* of local compute kernels and system-level communication infrastructure---a principle that motivates ongoing research in communication-avoiding algorithms, kernel fusion, and hardware interconnect design.

// ═════════════════════════════════════════════════════════════════════
// References
// ═════════════════════════════════════════════════════════════════════

#pagebreak()
#bibliography("references.bib", style: "ieee")
