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

KERNEL_ORDER = [
    "1_naive",
    "2_coalesced",
    "2_uncoalesced",
    "3_smem",
    "4_1d_blocktile",
    "5_2d_blocktile",
    "6_vectorized",
    "7_warptile",
    "cuBLAS",
]
KERNEL_LABELS = {
    "1_naive": "Naive",
    "2_coalesced": "Coalesced",
    "2_uncoalesced": "Uncoalesced",
    "3_smem": "Shared Mem",
    "4_1d_blocktile": "1D BlockTile",
    "5_2d_blocktile": "2D BlockTile",
    "6_vectorized": "Vectorized",
    "7_warptile": "WarpTile",
    "cuBLAS": "cuBLAS",
}
# Exclude uncoalesced from most plots (it's a negative example)
KERNELS_MAIN = [k for k in KERNEL_ORDER if k != "2_uncoalesced"]

COLORS = list(PAL.values())


def load():
    with open(DATA) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# Plot 1: Kernel GFLOPS comparison
# ══════════════════════════════════════════════════════════════════════════
def plot_kernel_gflops(data):
    perf = data["single_gpu"]["performance"]
    sizes = sorted(perf[KERNELS_MAIN[0]].keys(), key=int)
    kernels = KERNELS_MAIN

    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(sizes))
    n = len(kernels)
    w = 0.82 / n
    bar_colors = [PAL["blue"], PAL["teal"], PAL["green"], PAL["amber"],
                  PAL["orange"], PAL["red"], PAL["purple"], PAL["gray"]]

    for i, k in enumerate(kernels):
        vals = [perf[k][s] for s in sizes]
        ax.bar(
            x + i * w,
            vals,
            w,
            label=KERNEL_LABELS[k],
            color=bar_colors[i % len(bar_colors)],
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xlabel("Matrix Size ($M = N = K$)")
    ax.set_ylabel("GFLOPS")
    ax.set_title("Single-GPU GEMM Kernel Performance")
    ax.set_xticks(x + w * (n - 1) / 2)
    ax.set_xticklabels(sizes)
    ax.legend(ncol=2, loc="upper left", frameon=True)
    fig.savefig(FIGS / "kernel_gflops.pdf")
    plt.close(fig)
    print("  kernel_gflops.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 2: % of cuBLAS
# ══════════════════════════════════════════════════════════════════════════
def plot_cublas_pct(data):
    perf = data["single_gpu"]["performance"]
    sizes = sorted(perf["cuBLAS"].keys(), key=int)
    cublas = np.array([perf["cuBLAS"][s] for s in sizes])
    kernels = [k for k in KERNELS_MAIN if k != "cuBLAS"]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    line_colors = [PAL["blue"], PAL["teal"], PAL["green"], PAL["amber"],
                   PAL["orange"], PAL["red"], PAL["purple"]]
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
            color=line_colors[i % len(line_colors)],
        )

    ax.axhline(100, color=PAL["gray"], ls="--", alpha=0.5, lw=1, label="cuBLAS (100%)")
    ax.set_xlabel("Matrix Size ($M = N = K$)")
    ax.set_ylabel("\\% of cuBLAS Performance")
    ax.set_title("Kernel Performance Relative to cuBLAS")
    ax.legend(fontsize=8, ncol=2, frameon=True)
    ax.set_ylim(0, 110)
    fig.savefig(FIGS / "cublas_percentage.pdf")
    plt.close(fig)
    print("  cublas_percentage.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 3: Roofline
# ══════════════════════════════════════════════════════════════════════════
def plot_roofline(data):
    # H100 specs
    peak_gflops = 33500  # 33.5 TFLOPS FP32
    peak_bw = 3350  # 3.35 TB/s HBM3 → GB/s

    perf = data["single_gpu"]["performance"]
    size = "4096"

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Roofline curve
    oi = np.logspace(-1, 3, 500)
    roof = np.minimum(peak_gflops, peak_bw * oi)
    ax.loglog(oi, roof, "k-", lw=2.5, label="Roofline")

    ridge = peak_gflops / peak_bw
    ax.axvline(ridge, color="gray", ls=":", alpha=0.5)
    ax.text(
        ridge * 1.2,
        peak_gflops * 0.55,
        f"Ridge point\n({ridge:.1f} FLOP/B)",
        fontsize=8,
        color="gray",
    )

    kernels = KERNELS_MAIN
    markers = ["v", "^", "s", "D", "p", "h", "*", "o"]
    roof_colors = [PAL["blue"], PAL["teal"], PAL["green"], PAL["amber"],
                   PAL["orange"], PAL["red"], PAL["purple"], PAL["gray"]]

    for i, k in enumerate(kernels):
        gf = perf[k][size]
        eff_oi = gf / peak_bw
        ax.plot(
            eff_oi,
            gf,
            markers[i % len(markers)],
            color=roof_colors[i % len(roof_colors)],
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=KERNEL_LABELS[k],
            zorder=5,
        )

    ax.set_xlabel("Operational Intensity (FLOP/Byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(f"Roofline Model — H100 (size {size})")
    ax.set_xlim(0.08, 500)
    ax.set_ylim(100, peak_gflops * 1.3)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right", frameon=True)
    fig.savefig(FIGS / "roofline.pdf")
    plt.close(fig)
    print("  roofline.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 4: Strong scaling — time
# ══════════════════════════════════════════════════════════════════════════
def plot_strong_scaling(data):
    rows = data["multi_gpu"]["exp1"]["data"]
    sizes = sorted({r["M"] for r in rows})

    fig, ax = plt.subplots(figsize=(5, 3.8))
    for i, sz in enumerate(sizes):
        sub = sorted([r for r in rows if r["M"] == sz], key=lambda r: r["GPUs"])
        gpus = [r["GPUs"] for r in sub]
        times = [r["Total"] for r in sub]
        ax.plot(gpus, times, "o-", label=f"$N = {sz}$", lw=2, markersize=6, color=COLORS[i])
        t1 = times[0]
        ax.plot(gpus, [t1 / g for g in gpus], ":", color=COLORS[i], alpha=0.35, lw=1)

    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Strong Scaling — Column Parallel Forward")
    ax.legend(frameon=True)
    ax.set_xticks([1, 2, 4, 8])
    fig.savefig(FIGS / "strong_scaling.pdf")
    plt.close(fig)
    print("  strong_scaling.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 5: Strong scaling — parallel efficiency
# ══════════════════════════════════════════════════════════════════════════
def plot_strong_efficiency(data):
    rows = data["multi_gpu"]["exp1"]["data"]
    sizes = sorted({r["M"] for r in rows})

    fig, ax = plt.subplots(figsize=(5, 3.8))
    for i, sz in enumerate(sizes):
        sub = sorted([r for r in rows if r["M"] == sz], key=lambda r: r["GPUs"])
        t1 = sub[0]["Total"]
        gpus = [r["GPUs"] for r in sub]
        eff = [t1 / (g * r["Total"]) * 100 for g, r in zip(gpus, sub, strict=True)]
        ax.plot(gpus, eff, "o-", label=f"$N = {sz}$", lw=2, markersize=6, color=COLORS[i])

    ax.axhline(100, color=PAL["gray"], ls="--", alpha=0.4, lw=1)
    ax.set_xlabel("Number of GPUs")
    ax.set_ylabel("Parallel Efficiency (\\%)")
    ax.set_title("Strong Scaling Efficiency")
    ax.legend(frameon=True)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(0, 120)
    fig.savefig(FIGS / "strong_scaling_efficiency.pdf")
    plt.close(fig)
    print("  strong_scaling_efficiency.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 6: Weak scaling
# ══════════════════════════════════════════════════════════════════════════
def plot_weak_scaling(data):
    rows = sorted(data["multi_gpu"]["exp2"]["data"], key=lambda r: r["GPUs"])
    gpus = [r["GPUs"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

    ax1.plot(gpus, [r["Total"] for r in rows], "o-", lw=2, markersize=6, color=PAL["blue"])
    ax1.axhline(rows[0]["Total"], color=PAL["red"], ls="--", alpha=0.35, lw=1, label="Ideal")
    ax1.set_xlabel("Number of GPUs")
    ax1.set_ylabel("Total Time (ms)")
    ax1.set_title("Weak Scaling — Time")
    ax1.set_xticks(gpus)
    ax1.legend(frameon=True)

    ax2.plot(gpus, [r["GFLOPS"] for r in rows], "s-", lw=2, markersize=6, color=PAL["orange"])
    ideal_gf = rows[0]["GFLOPS"]
    ax2.plot(gpus, [ideal_gf * g for g in gpus], ":", color=PAL["red"], alpha=0.4, lw=1, label="Ideal linear")
    ax2.set_xlabel("Number of GPUs")
    ax2.set_ylabel("Aggregate GFLOPS")
    ax2.set_title("Weak Scaling — Throughput")
    ax2.set_xticks(gpus)
    ax2.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(FIGS / "weak_scaling.pdf")
    plt.close(fig)
    print("  weak_scaling.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 7: Comm vs Compute — by matrix size (exp3)
# ══════════════════════════════════════════════════════════════════════════
def plot_comm_compute_size(data):
    rows = sorted(data["multi_gpu"]["exp3"]["data"], key=lambda r: r["Size"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sizes = [str(r["Size"]) for r in rows]
    gemm = [r["GEMM"] for r in rows]
    comm = [r["Comm"] for r in rows]
    x = np.arange(len(sizes))
    w = 0.35

    ax.bar(x - w / 2, gemm, w, label="GEMM", color=PAL["blue"], edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, comm, w, label="Communication", color=PAL["orange"], edgecolor="white", linewidth=0.4)
    for i, r in enumerate(rows):
        ax.text(
            i + w / 2,
            r["Comm"] + 0.05,
            f"{r['Ratio']:.2f}",
            ha="center",
            fontsize=8,
            color=PAL["gray"],
        )

    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Compute vs Communication (8 GPUs)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(frameon=True)
    fig.savefig(FIGS / "comm_compute_ratio_size.pdf")
    plt.close(fig)
    print("  comm_compute_ratio_size.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 8: Comm/Compute ratio by kernel (exp4) — KEY RESULT
# ══════════════════════════════════════════════════════════════════════════
def plot_comm_compute_kernel(data):
    rows = data["multi_gpu"]["exp4"]["data"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [3, 2]})

    kernels = [r["Kernel"] for r in rows]
    labels = [KERNEL_LABELS.get(r["Kernel"], r["Kernel"]) for r in rows]
    gemm = [r["GEMM"] for r in rows]
    comm = [r["Comm"] for r in rows]
    ratio = [r["Ratio"] for r in rows]

    y = np.arange(len(kernels))
    ax1.barh(y, gemm, 0.6, label="GEMM", color=PAL["blue"], edgecolor="white", linewidth=0.4)
    ax1.barh(y, comm, 0.6, left=gemm, label="Communication", color=PAL["orange"], edgecolor="white", linewidth=0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Compute vs Communication per Kernel")
    ax1.legend(loc="lower right", fontsize=8, frameon=True)
    ax1.invert_yaxis()

    # Right: ratio with a sequential blue→red gradient
    norm_ratio = np.array(ratio) / max(ratio)
    ratio_colors = plt.cm.YlOrRd(norm_ratio * 0.8 + 0.1)
    ax2.barh(y, ratio, 0.6, color=ratio_colors, edgecolor="white", linewidth=0.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Comm / Compute Ratio")
    ax2.set_title("Communication Bottleneck")
    ax2.invert_yaxis()
    for i, v in enumerate(ratio):
        ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIGS / "comm_compute_ratio_kernel.pdf")
    plt.close(fig)
    print("  comm_compute_ratio_kernel.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 9: MLP forward / backward (exp5)
# ══════════════════════════════════════════════════════════════════════════
def plot_mlp(data):
    rows = sorted(data["multi_gpu"]["exp5"]["data"], key=lambda r: r["M"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sizes = [str(r["M"]) for r in rows]
    fwd = [r["Fwd"] for r in rows]
    bwd = [r["Bwd"] for r in rows]
    x = np.arange(len(sizes))

    ax.bar(x, fwd, 0.5, label="Forward", color=PAL["blue"], edgecolor="white", linewidth=0.4)
    ax.bar(x, bwd, 0.5, bottom=fwd, label="Backward", color=PAL["red"], edgecolor="white", linewidth=0.4)
    for i, r in enumerate(rows):
        ratio = r["Bwd"] / r["Fwd"]
        ax.text(
            i,
            r["Fwd"] + r["Bwd"] + 0.2,
            f"{ratio:.1f}\N{MULTIPLICATION SIGN}",
            ha="center",
            fontsize=8.5,
            color=PAL["gray"],
        )

    ax.set_xlabel("Matrix Size ($M = H = N$)")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Parallel MLP Timing (8 GPUs)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(frameon=True)
    fig.savefig(FIGS / "mlp_fwd_bwd.pdf")
    plt.close(fig)
    print("  mlp_fwd_bwd.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Plot 10: Overlap comparison (exp6)
# ══════════════════════════════════════════════════════════════════════════
def plot_overlap(data):
    rows = sorted(data["multi_gpu"]["exp6"]["data"], key=lambda r: r["Size"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sizes = [str(r["Size"]) for r in rows]
    no_ovlp = [r["NoOvlp"] for r in rows]
    ovlp = [r["Overlap"] for r in rows]
    x = np.arange(len(sizes))
    w = 0.3

    ax.bar(x - w / 2, no_ovlp, w, label="No Overlap", color=PAL["blue"], edgecolor="white", linewidth=0.4)
    ax.bar(x + w / 2, ovlp, w, label="Overlap (4 chunks)", color=PAL["green"], edgecolor="white", linewidth=0.4)
    for i, r in enumerate(rows):
        ax.text(
            i + w / 2,
            r["Overlap"] + 0.05,
            f"{r['Speedup']:.2f}\N{MULTIPLICATION SIGN}",
            ha="center",
            fontsize=8.5,
            color=PAL["gray"],
        )

    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Communication-Compute Overlap (8 GPUs)")
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.legend(frameon=True)
    fig.savefig(FIGS / "overlap_comparison.pdf")
    plt.close(fig)
    print("  overlap_comparison.pdf")


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
    print(f"All figures saved to {FIGS}/")


if __name__ == "__main__":
    main()
