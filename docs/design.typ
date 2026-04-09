// ═══════════════════════════════════════════════════════════════════════
// Software Design Document — Tensor Parallel GEMM
// ═══════════════════════════════════════════════════════════════════════

#set document(title: "Software Design: Tensor Parallel GEMM")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading: set block(above: 1.4em, below: 0.8em)

// ── Colors ───────────────────────────────────────────────────────────
#let blue = rgb("#3b82f6")
#let green = rgb("#22c55e")
#let orange = rgb("#f97316")
#let purple = rgb("#a855f7")
#let gray = rgb("#6b7280")
#let lightgray = rgb("#f3f4f6")
#let red = rgb("#ef4444")

// ── Box helpers ──────────────────────────────────────────────────────
#let component-box(title, body, color: blue) = {
  block(
    width: 100%,
    inset: 8pt,
    radius: 4pt,
    stroke: color + 1.2pt,
    fill: color.lighten(92%),
  )[
    #text(weight: "bold", fill: color)[#title] \
    #text(size: 9pt)[#body]
  ]
}

#let code-block(body) = {
  block(
    width: 100%,
    inset: 8pt,
    radius: 3pt,
    fill: lightgray,
  )[#text(font: "DejaVu Sans Mono", size: 8pt)[#body]]
}

// ── Title ────────────────────────────────────────────────────────────
#align(center)[
  #text(18pt, weight: "bold")[Software Design Document] \
  #v(0.3em)
  #text(13pt)[Tensor Parallel GEMM on Multi-GPU Systems] \
  #v(0.3em)
  #text(10pt, fill: gray)[SC4064 GPU Programming -- Aryan Jain, Fanyi Pu, Ze Hong Maxwell Au]
]

#v(1em)

// ═════════════════════════════════════════════════════════════════════
= Design Overview
// ═════════════════════════════════════════════════════════════════════

This document describes the software architecture of the Tensor Parallel GEMM project. The system has three layers: (1) a *kernel abstraction layer* with pluggable GEMM implementations, (2) a *CUDA resource layer* with RAII wrappers, and (3) a *tensor parallelism layer* that distributes computation across GPUs.

== Architecture Diagram

#figure(
  block(width: 100%, inset: 12pt)[
    #set text(size: 8.5pt)
    // ── Top: Benchmark / Application Layer ──
    #align(center)[
      #block(width: 85%, inset: 8pt, radius: 4pt, stroke: gray + 1pt, fill: gray.lighten(90%))[
        #align(center)[#text(weight: "bold", fill: gray)[Application Layer]]
        #v(4pt)
        #grid(
          columns: (1fr, 1fr),
          gutter: 8pt,
          block(inset: 6pt, radius: 3pt, stroke: gray + 0.5pt, fill: white)[
            `bench_single_gpu` \ Single-GPU benchmark
          ],
          block(inset: 6pt, radius: 3pt, stroke: gray + 0.5pt, fill: white)[
            `bench_multi_gpu` \ Multi-GPU scaling experiments
          ],
        )
      ]
    ]

    #v(6pt)
    #align(center)[#text(size: 16pt)[#sym.arrow.b]]
    #v(4pt)

    // ── Middle: Tensor Parallelism Layer ──
    #align(center)[
      #block(width: 85%, inset: 8pt, radius: 4pt, stroke: purple + 1pt, fill: purple.lighten(92%))[
        #align(center)[#text(weight: "bold", fill: purple)[Tensor Parallelism Layer]]
        #v(4pt)
        #grid(
          columns: (1fr, 1fr, 1fr),
          gutter: 6pt,
          block(inset: 5pt, radius: 3pt, stroke: purple + 0.5pt, fill: white)[
            *Column Parallel* \ Forward + Backward \ `AllGather` / `AllReduce`
          ],
          block(inset: 5pt, radius: 3pt, stroke: purple + 0.5pt, fill: white)[
            *Row Parallel* \ Forward + Backward \ `AllReduce`
          ],
          block(inset: 5pt, radius: 3pt, stroke: purple + 0.5pt, fill: white)[
            *MLP Block* \ Col $arrow.r$ Row compose \ Overlap pipelining
          ],
        )
      ]
    ]

    #v(6pt)
    #align(center)[#text(size: 16pt)[#sym.arrow.b]]
    #v(4pt)

    // ── Bottom: Two sub-layers side by side ──
    #grid(
      columns: (1fr, 1fr),
      gutter: 10pt,

      // Left: Kernel Layer
      block(inset: 8pt, radius: 4pt, stroke: blue + 1pt, fill: blue.lighten(92%))[
        #align(center)[#text(weight: "bold", fill: blue)[Kernel Abstraction Layer]]
        #v(4pt)
        #block(inset: 5pt, radius: 3pt, stroke: blue + 0.5pt, fill: white)[
          `GemmKernel` (abstract base) \
          #h(1em) #sym.arrow.r `name()`, `launch()` \
          #h(1em) #sym.arrow.r `needs_cublas()`
        ]
        #v(4pt)
        #block(inset: 5pt, radius: 3pt, stroke: blue + 0.5pt, fill: white)[
          `KernelRegistry` (singleton) \
          #h(1em) #sym.arrow.r `add()`, `get()`, `all()`
        ]
        #v(4pt)
        #grid(
          columns: (1fr, 1fr),
          gutter: 4pt,
          block(inset: 4pt, radius: 2pt, fill: blue.lighten(80%), stroke: none)[
            #text(size: 7.5pt)[`NaiveKernel`\ `CoalescedKernel`\ `SmemTilingKernel`\ `BlockTile1DKernel`]
          ],
          block(inset: 4pt, radius: 2pt, fill: blue.lighten(80%), stroke: none)[
            #text(size: 7.5pt)[`BlockTile2DKernel`\ `VectorizedKernel`\ `WarpTileKernel`\ `CublasKernel`]
          ],
        )
      ],

      // Right: CUDA Resource Layer
      block(inset: 8pt, radius: 4pt, stroke: green + 1pt, fill: green.lighten(92%))[
        #align(center)[#text(weight: "bold", fill: green)[CUDA Resource Layer (RAII)]]
        #v(4pt)
        #grid(
          columns: (1fr, 1fr),
          gutter: 4pt,
          block(inset: 5pt, radius: 3pt, stroke: green + 0.5pt, fill: white)[
            `CudaMemory<T>` \ Device allocation
          ],
          block(inset: 5pt, radius: 3pt, stroke: green + 0.5pt, fill: white)[
            `DeviceMatrix` \ 2D matrix wrapper
          ],
          block(inset: 5pt, radius: 3pt, stroke: green + 0.5pt, fill: white)[
            `CudaStream` \ Stream lifecycle
          ],
          block(inset: 5pt, radius: 3pt, stroke: green + 0.5pt, fill: white)[
            `CudaEvent` \ Timing & sync
          ],
          block(inset: 5pt, radius: 3pt, stroke: green + 0.5pt, fill: white)[
            `CublasHandle` \ cuBLAS context
          ],
          block(inset: 5pt, radius: 3pt, stroke: green + 0.5pt, fill: white)[
            `GpuTimer` \ Benchmark timing
          ],
        )
      ],
    )
  ],
  caption: [System architecture. The three layers are loosely coupled: kernels are pluggable via the registry, RAII wrappers manage GPU resources, and the tensor parallelism layer composes both.],
) <fig:arch>

// ═════════════════════════════════════════════════════════════════════
= Kernel Abstraction Layer
// ═════════════════════════════════════════════════════════════════════

The kernel layer provides a uniform interface for all GEMM implementations, from naive baselines to cuBLAS.

== Class Hierarchy

#figure(
  block(width: 90%, inset: 12pt)[
    #set text(size: 8.5pt)

    // Abstract base
    #align(center)[
      #block(width: 55%, inset: 8pt, radius: 4pt, stroke: blue + 1.5pt, fill: blue.lighten(92%))[
        #align(center)[
          #text(weight: "bold", size: 10pt)[`GemmKernel`] #text(fill: gray)[ (abstract)] \
          #line(length: 100%, stroke: 0.5pt + gray)
          #v(2pt)
          `+ name() -> const char*` #text(fill: red)[ = 0] \
          `+ launch(A, B, C, M, N, K, stream)` #text(fill: red)[ = 0] \
          `+ needs_cublas() -> bool` \
          `+ set_cublas_handle(handle)` \
        ]
      ]
    ]

    #v(4pt)
    #align(center)[
      #grid(
        columns: (1fr,) * 4,
        gutter: 4pt,
        ..range(4).map(_ => align(center)[#text(14pt)[#sym.triangle.b]])
      )
    ]
    #v(2pt)

    // Concrete classes
    #grid(
      columns: (1fr, 1fr, 1fr, 1fr),
      gutter: 6pt,
      block(inset: 5pt, radius: 3pt, stroke: blue + 0.8pt, fill: white)[
        #text(weight: "bold")[`NaiveKernel`] \
        Block: 32$times$32 \
        1 elem/thread
      ],
      block(inset: 5pt, radius: 3pt, stroke: blue + 0.8pt, fill: white)[
        #text(weight: "bold")[`SmemTiling`\ `Kernel`] \
        Tile: 32$times$32 \
        Shared memory
      ],
      block(inset: 5pt, radius: 3pt, stroke: blue + 0.8pt, fill: white)[
        #text(weight: "bold")[`WarpTile`\ `Kernel`] \
        BM$=$BN$=$128 \
        Warp-level tiling
      ],
      block(inset: 5pt, radius: 3pt, stroke: blue + 0.8pt, fill: white)[
        #text(weight: "bold")[`CublasKernel`] \
        cuBLAS `sgemm` \
        Tensor Core path
      ],
    )
    #v(4pt)
    #align(center)[#text(size: 8pt, fill: gray)[+ CoalescedKernel, BlockTile1DKernel, BlockTile2DKernel, VectorizedKernel]]
  ],
  caption: [Kernel class hierarchy. Each concrete kernel encapsulates its CUDA `__global__` function, block/grid configuration, and launch parameters. The abstract base provides a uniform `launch()` interface for all consumers.],
) <fig:class>

== Kernel Registry

The `KernelRegistry` is a singleton that manages kernel instances. Each kernel file self-registers via a static initializer:

#code-block[
```
// In naive.cu
class NaiveKernel : public GemmKernel { ... };

namespace { bool reg = [] {
    KernelRegistry::add(std::make_unique<NaiveKernel>());
    return true;
}(); }
```
]

This eliminates the parallel arrays (`kGemmLaunchFns`, `kGemmStreamFns`, `kGemmKernelNames`) that previously required manual synchronization. Adding a new kernel requires only creating the `.cu` file---no header modifications.

#figure(
  block(width: 80%, inset: 10pt)[
    #set text(size: 8.5pt)
    #grid(
      columns: (1fr, auto, 1fr),
      gutter: 8pt,

      // Before
      block(inset: 8pt, radius: 4pt, stroke: red + 1pt, fill: red.lighten(95%))[
        #align(center)[#text(weight: "bold", fill: red)[Before (fragile)]]
        #v(4pt)
        `enum GemmKernel { ... };` \
        `kGemmKernelNames[8]` \
        `kGemmLaunchFns[8]` \
        `kGemmStreamFns[8]` \
        #v(2pt)
        #text(fill: red, size: 8pt)[#sym.excl Arrays must stay in sync \ nullptr sentinel for cuBLAS \ Adding kernel = 4 edits]
      ],

      align(center + horizon)[#text(20pt)[#sym.arrow.r]],

      // After
      block(inset: 8pt, radius: 4pt, stroke: green + 1pt, fill: green.lighten(95%))[
        #align(center)[#text(weight: "bold", fill: green)[After (extensible)]]
        #v(4pt)
        `class GemmKernel { ... };` \
        `KernelRegistry::add(...)` \
        `KernelRegistry::get(id)` \
        `KernelRegistry::all()` \
        #v(2pt)
        #text(fill: green, size: 8pt)[#sym.checkmark Self-registering \ #sym.checkmark Polymorphic dispatch \ #sym.checkmark Adding kernel = 1 file]
      ],
    )
  ],
  caption: [Comparison of the old dispatch mechanism (parallel arrays + enum) versus the new registry-based approach. The new design is self-contained per kernel with no cross-file synchronization needed.],
) <fig:before_after>

// ═════════════════════════════════════════════════════════════════════
= CUDA Resource Layer
// ═════════════════════════════════════════════════════════════════════

All GPU resources are managed via RAII wrappers, ensuring deterministic cleanup and preventing leaks.

== Resource Ownership Model

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 6pt,
    align: (left, left, left, left),
    stroke: none,
    table.hline(),
    table.header([*Class*], [*Manages*], [*Acquire*], [*Release*]),
    table.hline(stroke: 0.5pt),
    [`CudaMemory<T>`], [Device allocation], [`cudaMalloc`], [`cudaFree`],
    [`DeviceMatrix`], [2D float matrix], [`CudaMemory<float>`], [Delegated],
    [`CudaStream`], [CUDA stream], [`cudaStreamCreate`], [`cudaStreamDestroy`],
    [`CudaEvent`], [CUDA event], [`cudaEventCreate`], [`cudaEventDestroy`],
    [`CublasHandle`], [cuBLAS context], [`cublasCreate`], [`cublasDestroy`],
    table.hline(),
  ),
  caption: [RAII resource wrappers. All classes are move-only (deleted copy constructor/assignment). Destruction is deterministic via C++ destructor ordering.],
) <tab:raii>

Key design decisions:
- *Move-only semantics*: All wrappers delete copy operations and implement move constructor/assignment. This prevents accidental double-free.
- *Implicit conversion*: `CudaStream`, `CudaEvent`, and `CublasHandle` provide `operator T()` for seamless interop with C CUDA APIs.
- *`DeviceMatrix` composition*: wraps `CudaMemory<float>` + dimensions, providing `init_random()`, `zero()`, and host transfer methods.

// ═════════════════════════════════════════════════════════════════════
= Tensor Parallelism Layer
// ═════════════════════════════════════════════════════════════════════

The TP layer implements distributed linear layers following Megatron-LM.

== Data Flow: Parallel MLP Block

#figure(
  block(width: 95%, inset: 10pt)[
    #set text(size: 8.5pt)

    // Forward pass
    #block(inset: 8pt, radius: 4pt, stroke: purple + 1pt, fill: purple.lighten(95%))[
      #text(weight: "bold", fill: purple)[Forward Pass]
      #v(4pt)
      #grid(
        columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
        gutter: 0pt,
        align: center + horizon,
        block(inset: 4pt, radius: 3pt, stroke: gray, fill: white)[
          $X$ \ (replicated)
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: blue, fill: blue.lighten(92%))[
          *GEMM* \ $H_i = X W_(1,i)$
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: gray, fill: white)[
          $H_i$ \ (local shard)
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: blue, fill: blue.lighten(92%))[
          *GEMM* \ $Y_i = H_i W_(2,i)$
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: orange, fill: orange.lighten(90%))[
          *AllReduce* \ $Y = sum Y_i$
        ],
      )
      #v(4pt)
      #align(center)[#text(size: 8pt, fill: gray)[Column Parallel (no comm) #h(4em) Row Parallel (1 AllReduce)]]
    ]

    #v(8pt)

    // Backward pass
    #block(inset: 8pt, radius: 4pt, stroke: red.lighten(20%) + 1pt, fill: red.lighten(95%))[
      #text(weight: "bold", fill: red.lighten(20%))[Backward Pass]
      #v(4pt)
      #grid(
        columns: (1fr, auto, 1fr, auto, 1fr, auto, 1fr, auto, 1fr),
        gutter: 0pt,
        align: center + horizon,
        block(inset: 4pt, radius: 3pt, stroke: gray, fill: white)[
          $dif Y$ \ (replicated)
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: blue, fill: blue.lighten(92%))[
          *GEMM* $times 2$ \ $dif W_(2,i), dif H_i$
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: gray, fill: white)[
          $dif H_i$ \ (local)
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: blue, fill: blue.lighten(92%))[
          *GEMM* $times 2$ \ $dif W_(1,i), dif X_i$
        ],
        [#sym.arrow.r],
        block(inset: 4pt, radius: 3pt, stroke: orange, fill: orange.lighten(90%))[
          *AllReduce* \ $dif X = sum dif X_i$
        ],
      )
      #v(4pt)
      #align(center)[#text(size: 8pt, fill: gray)[Row Backward (no comm) #h(4em) Column Backward (1 AllReduce)]]
    ]
  ],
  caption: [Data flow for a parallel MLP block (forward and backward). Blue boxes are local GEMM operations dispatched through `GemmKernel::launch()`. Orange boxes are NCCL collectives. The key design insight is that column $arrow.r$ row composition requires only *one AllReduce per pass*.],
) <fig:mlp_flow>

== Communication-Compute Overlap

For the row-parallel forward, we support chunked pipelining:

#figure(
  block(width: 85%, inset: 10pt)[
    #set text(size: 8.5pt)

    // No overlap
    #block(inset: 6pt, radius: 3pt, stroke: gray + 0.5pt, fill: white)[
      #text(weight: "bold")[Sequential (no overlap)]
      #v(4pt)
      #grid(
        columns: (3fr, 3fr, 0.5fr),
        gutter: 0pt,
        block(inset: 4pt, fill: blue.lighten(75%), radius: 2pt)[
          #align(center)[GEMM (full)]
        ],
        block(inset: 4pt, fill: orange.lighten(75%), radius: 2pt)[
          #align(center)[AllReduce (full)]
        ],
        [],
      )
    ]

    #v(8pt)

    // With overlap
    #block(inset: 6pt, radius: 3pt, stroke: green + 0.5pt, fill: white)[
      #text(weight: "bold")[Pipelined (4 chunks)]
      #v(4pt)
      #text(size: 7.5pt, fill: gray)[Compute stream:]
      #grid(
        columns: (1fr, 1fr, 1fr, 1fr, 2fr),
        gutter: 2pt,
        block(inset: 3pt, fill: blue.lighten(75%), radius: 2pt)[#align(center)[G1]],
        block(inset: 3pt, fill: blue.lighten(75%), radius: 2pt)[#align(center)[G2]],
        block(inset: 3pt, fill: blue.lighten(75%), radius: 2pt)[#align(center)[G3]],
        block(inset: 3pt, fill: blue.lighten(75%), radius: 2pt)[#align(center)[G4]],
        [],
      )
      #v(2pt)
      #text(size: 7.5pt, fill: gray)[Comm stream:]
      #grid(
        columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
        gutter: 2pt,
        [],
        block(inset: 3pt, fill: orange.lighten(75%), radius: 2pt)[#align(center)[A1]],
        block(inset: 3pt, fill: orange.lighten(75%), radius: 2pt)[#align(center)[A2]],
        block(inset: 3pt, fill: orange.lighten(75%), radius: 2pt)[#align(center)[A3]],
        block(inset: 3pt, fill: orange.lighten(75%), radius: 2pt)[#align(center)[A4]],
        [],
      )
      #v(2pt)
      #text(size: 7.5pt, fill: gray)[`cudaEvent` synchronization between streams at chunk boundaries]
    ]
  ],
  caption: [Communication-compute overlap via chunked pipelining. The $M$ dimension is split into chunks; each chunk's `AllReduce` (Ai) is overlapped with the next chunk's GEMM (Gi+1) using dual CUDA streams and event-based synchronization.],
) <fig:overlap>

// ═════════════════════════════════════════════════════════════════════
= File Organization
// ═════════════════════════════════════════════════════════════════════

#figure(
  block(width: 80%, inset: 10pt)[
    #set text(font: "DejaVu Sans Mono", size: 8pt)
    ```
    src/
    +-- kernels/
    |   +-- gemm_kernel.cuh          # Abstract base class
    |   +-- kernel_registry.cuh      # Singleton registry
    |   +-- naive.cu                  # NaiveKernel
    |   +-- coalesced.cu             # CoalescedKernel
    |   +-- smem_tiling.cu           # SmemTilingKernel
    |   +-- blocktile_1d.cu          # BlockTile1DKernel
    |   +-- blocktile_2d.cu          # BlockTile2DKernel
    |   +-- vectorized.cu            # VectorizedKernel
    |   +-- warptile.cu              # WarpTileKernel
    |   +-- cublas.cu                # CublasKernel
    +-- tensor_parallel/
    |   +-- tensor_parallel.cuh      # TP layer declarations
    |   +-- tensor_parallel.cu       # TP layer implementation
    +-- benchmark/
    |   +-- bench_single_gpu.cu      # Single-GPU experiments
    |   +-- bench_multi_gpu.cu       # Multi-GPU experiments
    +-- utils/
        +-- cuda_utils.cuh           # Error macros, timing, verification
        +-- cuda_raii.cuh            # RAII wrappers
        +-- device_matrix.cuh        # DeviceMatrix
        +-- nccl_utils.cuh           # NCCL error checking
    ```
  ],
  caption: [Project file organization after refactoring. Numbered prefixes are removed from kernel files; each file contains a self-registering kernel class.],
) <fig:files>

// ═════════════════════════════════════════════════════════════════════
= Design Decisions
// ═════════════════════════════════════════════════════════════════════

== Why Virtual Dispatch for Kernels?

The overhead of a virtual function call ($tilde$1 ns) is negligible compared to GEMM kernel execution ($tilde$0.5--1000 ms). This trade-off buys us:
- *Open-closed principle*: new kernels can be added without modifying existing code.
- *Encapsulation*: tile sizes, block dimensions, and launch configuration are private to each kernel class.
- *Type safety*: the enum + function-pointer-array pattern had no compile-time guarantee that arrays stayed in sync.

== Why Self-Registering over Explicit Enum?

The old design required touching 4 places to add a kernel: (1) enum value, (2) name string, (3) launch function pointer, (4) stream function pointer. A self-registering pattern puts all kernel metadata in one file:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 6pt,
    stroke: none,
    table.hline(),
    table.header([*Operation*], [*Old (enum + arrays)*], [*New (registry)*]),
    table.hline(stroke: 0.5pt),
    [Add a kernel], [4 files, 4 edits], [1 file, self-contained],
    [Remove a kernel], [4 files, careful deletion], [Delete 1 file],
    [Rename a kernel], [Update enum + name string], [Change `name()` return],
    [Iterate all kernels], [`for (int i = 0; i < COUNT; i++)`], [`for (auto& k : KernelRegistry::all())`],
    table.hline(),
  ),
  caption: [Maintenance comparison between the enum-based and registry-based approaches.],
)

== RAII as Error Prevention

Every CUDA resource follows the RAII pattern:

#code-block[
```
// Old pattern (error-prone):
float* d_ptr;
cudaMalloc(&d_ptr, size);
// ... if exception or early return, d_ptr leaks
cudaFree(d_ptr);

// New pattern (leak-proof):
CudaMemory<float> d_ptr(count);
// Automatically freed when scope exits
```
]

This is enforced by making all wrappers *move-only* (deleted copy), preventing accidental aliasing of GPU resources.
