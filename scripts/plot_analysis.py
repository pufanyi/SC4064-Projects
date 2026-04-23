#!/usr/bin/env python3
"""Generate publication-quality analysis plots from benchmark_results.json."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "benchmark_results.json"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 8.5,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "#cccccc",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "axes.grid": True,
        "grid.alpha": 0.2,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 2,
        "lines.markersize": 6,
    }
)

# Cohesive palette — colourblind-friendly, print-safe
PAL = {
    "blue": "#2563eb",
    "orange": "#ea580c",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "teal": "#0891b2",
    "pink": "#db2777",
    "gray": "#6b7280",
    "amber": "#d97706",
}
COLORS = list(PAL.values())

# Kernel names match bench_single_gpu.cu / KernelRegistry output in JSON
KERNEL_ORDER = [
    "Naive",
    "Coalesced",
    "Uncoalesced",
    "SmemTiling",
    "BlockTile1D",
    "BlockTile2D",
    "Vectorized",
    "WarpTile",
    "cuBLAS",
]
KERNEL_LABELS = {
    "Naive": "Naive",
    "Coalesced": "Coalesced",
    "Uncoalesced": "Uncoalesced",
    "SmemTiling": "Shared Mem",
    "BlockTile1D": "1D BlockTile",
    "BlockTile2D": "2D BlockTile",
    "Vectorized": "Vectorized",
    "WarpTile": "WarpTile",
    "cuBLAS": "cuBLAS",
}
# Uncoalesced is a negative example — drop from multi-kernel overview plots
KERNELS_MAIN = [k for k in KERNEL_ORDER if k != "Uncoalesced"]

KERNEL_COLORS = [
    PAL["blue"],
    PAL["teal"],
    PAL["green"],
    PAL["amber"],
    PAL["orange"],
    PAL["red"],
    PAL["purple"],
    "#111827",  # near-black for cuBLAS so it stays readable over others
]

TRANSPORT_STYLE = {
    "auto": {"label": "Auto (NCCL default)", "color": PAL["blue"], "marker": "o"},
    "ib": {"label": "InfiniBand", "color": PAL["green"], "marker": "s"},
    "ring": {"label": "IB + Ring", "color": PAL["purple"], "marker": "^"},
    "tcp": {"label": "TCP socket", "color": PAL["red"], "marker": "D"},
}


def load():
    with open(DATA) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# Plot 1: Kernel GFLOPS comparison (grouped bars)
# ══════════════════════════════════════════════════════════════════════════
def plot_kernel_gflops(data):
    perf = data["single_gpu"]["performance"]
    sizes = sorted(perf[KERNELS_MAIN[0]].keys(), key=int)
    kernels = KERNELS_MAIN

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    x = np.arange(len(sizes))
    n = len(kernels)
    w = 0.82 / n

    for i, k in enumerate(kernels):
        vals = [perf[k][s] for s in sizes]
        ax.bar(
            x + i * w,
            vals,
            w,
            label=KERNEL_LABELS[k],
            color=KERNEL_COLORS[i % len(KERNEL_COLORS)],
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xlabel("Matrix Size ($M = N = K$)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Single-GPU GEMM Kernel Performance (H100)")
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(sizes)
    ax.legend(ncol=2, loc="upper left", frameon=True)
    fig.savefig(FIGS / "kernel_gflops.pdf")
    plt.close(fig)
    print("  kernel_gflops.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2: kernel performance as % of cuBLAS
# ══════════════════════════════════════════════════════════════════════════
def plot_cublas_pct(data):
    perf = data["single_gpu"]["performance"]
    sizes = sorted(perf["cuBLAS"].keys(), key=int)
    cublas = np.array([perf["cuBLAS"][s] for s in sizes])
    kernels = [k for k in KERNELS_MAIN if k != "cuBLAS"]

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    for i, k in enumerate(kernels):
        vals = np.array([perf[k][s] for s in sizes])
        pct = vals / cublas * 100
        ax.plot(
            [int(s) for s in sizes],
            pct,
            "o-",
            label=KERNEL_LABELS[k],
            linewidth=2,
            markersize=5,
            color=KERNEL_COLORS[i % len(KERNEL_COLORS)],
        )

    ax.axhline(100, color=PAL["gray"], ls="--", alpha=0.5, lw=1, label="cuBLAS (100%)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix Size ($M = N = K$)")
    ax.set_ylabel("% of cuBLAS Performance")
    ax.set_title("Kernel Performance Relative to cuBLAS")
    ax.legend(fontsize=8, ncol=2, frameon=True)
    ax.set_ylim(0, 110)
    fig.savefig(FIGS / "cublas_percentage.pdf")
    plt.close(fig)
    print("  cublas_percentage.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3: Roofline with kernel data points
# ══════════════════════════════════════════════════════════════════════════
def plot_roofline(data):
    # H100 PCIe FP32 dense FMA peak = 132 SM x 128 FP32 ALU x 1.98 GHz x 2 (FMA)
    # ~= 67 TFLOPS.  The device-query in bench_single_gpu prints 33.5 TFLOPS
    # because it doesn't double-count the multiply-add; the real SGEMM ceiling
    # is 2x that.
    peak_gflops = 67000
    peak_bw = 3350  # 3.35 TB/s HBM3

    perf = data["single_gpu"]["performance"]
    size = "4096"

    # SGEMM operational intensity at size N:
    #   ops = 2 N^3 FMA-flops;  bytes = 3 N^2 x 4 B  (A, B, C in fp32)
    N = int(size)
    oi_gemm = (2.0 * N**3) / (3.0 * N**2 * 4)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    oi = np.logspace(-1, 3, 500)
    roof = np.minimum(peak_gflops, peak_bw * oi)
    ax.loglog(oi, roof, "k-", lw=2, label="Roofline (FP32 peak)")

    ridge = peak_gflops / peak_bw
    ax.axvline(ridge, color="gray", ls=":", alpha=0.5)
    ax.text(
        ridge,
        peak_gflops * 1.45,
        f"ridge\n{ridge:.0f} FLOP/B",
        fontsize=8,
        color="gray",
        ha="center",
        va="bottom",
    )

    ax.axvline(oi_gemm, color=PAL["blue"], ls=":", alpha=0.35, lw=1)
    ax.text(
        oi_gemm,
        peak_gflops * 1.45,
        f"SGEMM OI\n{oi_gemm:.0f} FLOP/B",
        fontsize=8,
        color=PAL["blue"],
        ha="center",
        va="bottom",
    )

    markers = ["v", "^", "s", "D", "p", "h", "*", "o"]
    # Slight horizontal jitter so markers at identical OI don't fully occlude.
    n = len(KERNELS_MAIN)
    jitter = np.logspace(-0.11, 0.11, n)
    for i, k in enumerate(KERNELS_MAIN):
        gf = perf[k][size]
        ax.plot(
            oi_gemm * jitter[i],
            gf,
            markers[i % len(markers)],
            color=KERNEL_COLORS[i % len(KERNEL_COLORS)],
            markersize=11,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=KERNEL_LABELS[k],
            zorder=5,
        )

    ax.set_xlabel("Operational Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(f"Roofline — H100 FP32, SGEMM at $N={size}$")
    ax.set_xlim(0.08, 1500)
    ax.set_ylim(100, peak_gflops * 2.0)
    ax.legend(
        fontsize=7.5, ncol=2, loc="lower right", frameon=True, facecolor="white", framealpha=0.95
    )
    fig.savefig(FIGS / "roofline.pdf")
    plt.close(fig)
    print("  roofline.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 4: Strong scaling — combined single-node (1-8, NVLink) + cross-node (16, IB)
# ══════════════════════════════════════════════════════════════════════════
def plot_strong_scaling(data):
    single = data["multi_gpu"]["exp1"]["data"]
    multi = data["multi_node"]["transports"]["auto"]["exp1"]["data"]

    single_sizes = sorted({r["M"] for r in single})

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for i, sz in enumerate(single_sizes):
        sn = sorted([r for r in single if r["M"] == sz], key=lambda r: r["GPUs"])
        mn = [r for r in multi if r["M"] == sz]

        gpus = [r["GPUs"] for r in sn] + [r["GPUs"] for r in mn]
        times = [r["Total"] for r in sn] + [r["Total"] for r in mn]

        ax.plot(
            gpus,
            times,
            "o-",
            label=f"$N = {sz}$",
            lw=1.8,
            markersize=6,
            color=COLORS[i],
        )

        # Ideal-scaling reference (T1 / G) -- same colour as the curve,
        # but dashed and muted so it stays a background guide.
        t1 = sn[0]["Total"]
        max_g = max(gpus)
        g_ref = np.array([1, max_g])
        ax.plot(g_ref, t1 / g_ref, linestyle=(0, (3, 2)), color=COLORS[i], alpha=0.65, lw=1.3)

    ax.axvline(8.5, color=PAL["gray"], ls="--", alpha=0.4, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels([1, 2, 4, 8, 16])
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Strong Scaling — Column-Parallel Forward")
    # Add a single entry in the legend so readers know what the dashed guides
    # mean, then let matplotlib autoscale so the ideal guides are fully visible.
    ax.plot(
        [],
        [],
        linestyle=(0, (3, 2)),
        color=PAL["gray"],
        alpha=0.75,
        lw=1.3,
        label="Ideal $T_1 / P$",
    )
    # Legend sits outside the axes to the right so it never clips the data lines.
    ax.legend(title="Matrix size", frameon=True, loc="center left", bbox_to_anchor=(1.02, 0.5))
    _, ytop = ax.get_ylim()
    ax.text(8.7, ytop * 0.45, "NVLink │ IB", fontsize=8.5, color=PAL["gray"], rotation=90, va="top")
    fig.savefig(FIGS / "strong_scaling.pdf")
    plt.close(fig)
    print("  strong_scaling.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5: Strong-scaling parallel efficiency (single + multi-node)
# ══════════════════════════════════════════════════════════════════════════
def plot_strong_efficiency(data):
    single = data["multi_gpu"]["exp1"]["data"]
    multi = data["multi_node"]["transports"]["auto"]["exp1"]["data"]
    single_sizes = sorted({r["M"] for r in single})

    fig, ax = plt.subplots(figsize=(5.6, 4))
    for i, sz in enumerate(single_sizes):
        sn = sorted([r for r in single if r["M"] == sz], key=lambda r: r["GPUs"])
        mn = [r for r in multi if r["M"] == sz]
        t1 = sn[0]["Total"]
        combined = sn + mn
        gpus = [r["GPUs"] for r in combined]
        eff = [t1 / (g * r["Total"]) * 100 for g, r in zip(gpus, combined, strict=True)]
        ax.plot(gpus, eff, "o-", label=f"$N = {sz}$", lw=1.8, markersize=6, color=COLORS[i])

    ax.axvline(8.5, color=PAL["gray"], ls="--", alpha=0.4, lw=1)
    ax.text(8.7, 125, "NVLink │ IB", fontsize=8.5, color=PAL["gray"], rotation=90, va="top")
    ax.axhline(100, color=PAL["gray"], ls="--", alpha=0.4, lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels([1, 2, 4, 8, 16])
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Strong Scaling Efficiency")
    ax.legend(title="Matrix size", frameon=True, loc="lower left")
    ax.set_ylim(0, 130)
    fig.savefig(FIGS / "strong_scaling_efficiency.pdf")
    plt.close(fig)
    print("  strong_scaling_efficiency.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 6: Weak scaling — one line per per-GPU tile size
# ══════════════════════════════════════════════════════════════════════════
def plot_weak_scaling(data):
    rows = data["multi_gpu"]["exp2"]["data"]
    tiles = sorted({r["M"] for r in rows})
    all_gpus = sorted({r["GPUs"] for r in rows})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    for i, tile in enumerate(tiles):
        sub = sorted([r for r in rows if r["M"] == tile], key=lambda r: r["GPUs"])
        gpus = [r["GPUs"] for r in sub]
        times = [r["Total"] for r in sub]
        gflops = [r["GFLOPS"] for r in sub]

        ax1.plot(
            gpus,
            times,
            "o-",
            label=f"tile = {tile}",
            color=COLORS[i],
            lw=1.8,
            markersize=5,
        )
        ax2.plot(gpus, gflops, "s-", label=f"tile = {tile}", color=COLORS[i], lw=1.8, markersize=5)

    # Ideal (flat time, linear throughput) reference on each, relative to tile=2048
    base = next(r for r in rows if r["M"] == tiles[0] and r["GPUs"] == 1)
    ax1.axhline(
        base["Total"],
        linestyle=(0, (3, 2)),
        color=PAL["gray"],
        alpha=0.75,
        lw=1.3,
        label="Ideal (tile=2048)",
    )
    ax2.plot(
        all_gpus,
        [base["GFLOPS"] * g for g in all_gpus],
        linestyle=(0, (3, 2)),
        color=PAL["gray"],
        alpha=0.75,
        lw=1.3,
        label="Ideal (tile=2048)",
    )

    for ax in (ax1, ax2):
        ax.set_xlabel("Number of GPUs")
        ax.set_xticks(all_gpus)

    ax1.set_ylabel("Total Time (ms)")
    ax1.set_title("Weak Scaling — Time")
    ax1.set_yscale("log")
    ax2.set_ylabel("Aggregate GFLOPS")
    ax2.set_title("Weak Scaling — Throughput")

    # Shared legend below both subplots so it never overlaps the data.
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS / "weak_scaling.pdf")
    plt.close(fig)
    print("  weak_scaling.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7: Comm vs Compute — by matrix size (exp3) at 8 GPUs
# ══════════════════════════════════════════════════════════════════════════
def plot_comm_compute_size(data):
    rows = sorted(data["multi_gpu"]["exp3"]["data"], key=lambda r: r["Size"])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    sizes = [str(r["Size"]) for r in rows]
    gemm = np.array([r["GEMM"] for r in rows])
    comm = np.array([r["Comm"] for r in rows])
    x = np.arange(len(sizes))
    w = 0.36

    ax.bar(x - w / 2, gemm, w, label="GEMM", color=PAL["blue"], edgecolor="white", linewidth=0.4)
    ax.bar(
        x + w / 2,
        comm,
        w,
        label="Communication",
        color=PAL["orange"],
        edgecolor="white",
        linewidth=0.4,
    )
    for i, r in enumerate(rows):
        y = max(r["GEMM"], r["Comm"])
        ax.text(i, y * 1.12, f"ratio {r['Ratio']:.2f}", ha="center", fontsize=8, color=PAL["gray"])

    ax.set_xlabel("Matrix Size ($M = N = K$)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Compute vs Communication (8 GPUs, NVLink)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_yscale("log")
    ax.legend(frameon=True, loc="upper left")
    fig.savefig(FIGS / "comm_compute_ratio_size.pdf")
    plt.close(fig)
    print("  comm_compute_ratio_size.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 8: Comm/Compute ratio by kernel (exp4) — two-panel
# ══════════════════════════════════════════════════════════════════════════
def plot_comm_compute_kernel(data):
    rows = data["multi_gpu"]["exp4"]["data"]
    # Drop Uncoalesced -- its 35 ms GEMM squashes the chart for everyone else.
    rows = [r for r in rows if r["Kernel"] != "Uncoalesced"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [3, 2]})

    labels = [KERNEL_LABELS.get(r["Kernel"], r["Kernel"]) for r in rows]
    gemm = [r["GEMM"] for r in rows]
    comm = [r["Comm"] for r in rows]
    ratio = [r["Ratio"] for r in rows]
    y = np.arange(len(rows))

    ax1.barh(y, gemm, 0.62, label="GEMM", color=PAL["blue"], edgecolor="white", linewidth=0.4)
    ax1.barh(
        y,
        comm,
        0.62,
        left=gemm,
        label="Communication",
        color=PAL["orange"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Compute vs Communication per Kernel")
    ax1.legend(loc="lower right", fontsize=8, frameon=True)
    ax1.invert_yaxis()

    norm = np.array(ratio) / max(ratio)
    ratio_colors = plt.cm.YlOrRd(norm * 0.8 + 0.1)
    ax2.barh(y, ratio, 0.62, color=ratio_colors, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Comm / Compute Ratio")
    ax2.set_title("Communication Bottleneck")
    ax2.invert_yaxis()
    for i, v in enumerate(ratio):
        ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8.5)

    fig.suptitle("Local GEMM Kernel vs Communication ($N = 4096$, 8 GPUs)", fontsize=11.5, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "comm_compute_ratio_kernel.pdf")
    plt.close(fig)
    print("  comm_compute_ratio_kernel.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9: MLP forward / backward (exp5)
# ══════════════════════════════════════════════════════════════════════════
def plot_mlp(data):
    rows = sorted(data["multi_gpu"]["exp5"]["data"], key=lambda r: r["M"])

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    sizes = [str(r["M"]) for r in rows]
    fwd = np.array([r["Fwd"] for r in rows])
    bwd = np.array([r["Bwd"] for r in rows])
    x = np.arange(len(sizes))

    ax.bar(x, fwd, 0.5, label="Forward", color=PAL["blue"], edgecolor="white", linewidth=0.4)
    ax.bar(
        x,
        bwd,
        0.5,
        bottom=fwd,
        label="Backward",
        color=PAL["red"],
        edgecolor="white",
        linewidth=0.4,
    )
    for i, r in enumerate(rows):
        ratio = r["Bwd"] / r["Fwd"]
        ax.text(
            i,
            r["Fwd"] + r["Bwd"] + (r["Fwd"] + r["Bwd"]) * 0.06,
            f"{ratio:.1f}\N{MULTIPLICATION SIGN}",
            ha="center",
            fontsize=8.5,
            color=PAL["gray"],
        )

    ax.set_xlabel("Matrix Size ($M = H = N$)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Parallel MLP Forward + Backward (8 GPUs)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_yscale("log")
    ax.legend(frameon=True, loc="upper left")
    fig.savefig(FIGS / "mlp_fwd_bwd.pdf")
    plt.close(fig)
    print("  mlp_fwd_bwd.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 10: Overlap vs No-Overlap (exp6)
# ══════════════════════════════════════════════════════════════════════════
def plot_overlap(data):
    rows = sorted(data["multi_gpu"]["exp6"]["data"], key=lambda r: r["Size"])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    sizes = [str(r["Size"]) for r in rows]
    no_ovlp = np.array([r["NoOvlp"] for r in rows])
    ovlp = np.array([r["Overlap"] for r in rows])
    x = np.arange(len(sizes))
    w = 0.33

    ax.bar(
        x - w / 2,
        no_ovlp,
        w,
        label="No overlap",
        color=PAL["blue"],
        edgecolor="white",
        linewidth=0.4,
    )
    ax.bar(
        x + w / 2,
        ovlp,
        w,
        label="Overlap (4 chunks)",
        color=PAL["green"],
        edgecolor="white",
        linewidth=0.4,
    )
    for i, r in enumerate(rows):
        top = max(r["NoOvlp"], r["Overlap"])
        ax.text(
            i,
            top * 1.1,
            f"{r['Speedup']:.2f}\N{MULTIPLICATION SIGN}",
            ha="center",
            fontsize=8.5,
            color=PAL["gray"],
        )

    ax.set_xlabel("Matrix Size ($M = N = K$)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Communication-Compute Overlap (Row Parallel, 8 GPUs)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_yscale("log")
    ax.legend(frameon=True, loc="upper left")
    fig.savefig(FIGS / "overlap_comparison.pdf")
    plt.close(fig)
    print("  overlap_comparison.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 11: Transport sweep — Comm time vs size for auto / IB / ring / TCP
# ══════════════════════════════════════════════════════════════════════════
def plot_transport_sweep(data):
    transports = data["multi_node"]["transports"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    for tag, style in TRANSPORT_STYLE.items():
        if tag not in transports:
            continue
        rows = sorted(transports[tag]["exp1"]["data"], key=lambda r: r["M"])
        sizes = [r["M"] for r in rows]
        comm = [r["Comm"] for r in rows]
        ax1.plot(
            sizes,
            comm,
            f"{style['marker']}-",
            label=style["label"],
            color=style["color"],
            lw=1.8,
            markersize=6,
        )

        r3 = sorted(transports[tag]["exp3"]["data"], key=lambda r: r["Size"])
        ax2.plot(
            [r["Size"] for r in r3],
            [r["Ratio"] for r in r3],
            f"{style['marker']}-",
            label=style["label"],
            color=style["color"],
            lw=1.8,
            markersize=6,
        )

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Matrix Size ($M = N = K$)")
    ax1.set_ylabel("AllGather Time (ms)")
    ax1.set_title("Transport Latency (16 GPUs, 2 nodes)")
    ax1.legend(frameon=True, loc="upper left")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.axhline(1.0, color=PAL["gray"], ls="--", alpha=0.45, lw=1)
    ax2.set_xlabel("Matrix Size ($M = N = K$)")
    ax2.set_ylabel("Comm / Compute Ratio")
    ax2.set_title("Communication Bottleneck by Transport")
    ax2.legend(frameon=True, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIGS / "transport_sweep.pdf")
    plt.close(fig)
    print("  transport_sweep.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
def main():
    data = load()
    print("Generating figures:")
    plot_kernel_gflops(data)
    plot_cublas_pct(data)
    plot_roofline(data)
    plot_strong_scaling(data)
    plot_strong_efficiency(data)
    plot_weak_scaling(data)
    plot_comm_compute_size(data)
    plot_comm_compute_kernel(data)
    plot_mlp(data)
    plot_overlap(data)
    plot_transport_sweep(data)
    print(f"All figures saved to {FIGS}/")


if __name__ == "__main__":
    main()
