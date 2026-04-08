#!/usr/bin/env python3
"""
Plot benchmark results from bench_single_gpu and bench_multi_gpu.

Usage:
    ./bench_single_gpu | tee results/single_gpu.txt
    ./bench_multi_gpu 8 7 | tee results/multi_gpu.txt
    python3 scripts/plot_results.py results/single_gpu.txt [results/multi_gpu.txt]
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_single_gpu(filename):
    """Parse the GFLOPS table from bench_single_gpu output."""
    kernels = {}
    sizes = []
    in_table = False

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if "Performance Benchmark" in line:
                in_table = True
                continue
            if in_table and line.startswith("Kernel"):
                sizes = [int(x) for x in line.split()[1:]]
                continue
            if in_table and line.startswith("---"):
                continue
            if in_table and line and not line.startswith("Done"):
                parts = line.split()
                name = parts[0]
                gflops = [float(x) for x in parts[1:]]
                kernels[name] = gflops
            if "Done" in line:
                break

    return sizes, kernels


def plot_gflops_comparison(sizes, kernels, outfile="results/gflops_comparison.png"):
    """Bar chart of GFLOPS across kernel optimizations."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(sizes))
    n = len(kernels)
    width = 0.8 / n

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    for i, (name, gflops) in enumerate(kernels.items()):
        ax.bar(x + i * width, gflops, width, label=name, color=colors[i])

    ax.set_xlabel("Matrix Size (M=N=K)", fontsize=12)
    ax.set_ylabel("GFLOPS", fontsize=12)
    ax.set_title("GEMM Kernel Performance Comparison", fontsize=14)
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_cublas_percentage(sizes, kernels, outfile="results/cublas_percentage.png"):
    """Line chart showing each kernel as % of cuBLAS performance."""
    if "cuBLAS" not in kernels:
        print("No cuBLAS data found, skipping percentage plot.")
        return

    cublas = np.array(kernels["cuBLAS"])
    fig, ax = plt.subplots(figsize=(12, 6))

    for name, gflops in kernels.items():
        if name == "cuBLAS":
            continue
        pct = np.array(gflops) / cublas * 100
        ax.plot(sizes, pct, "o-", label=name, linewidth=2, markersize=6)

    ax.axhline(y=100, color="red", linestyle="--", alpha=0.5, label="cuBLAS (100%)")
    ax.set_xlabel("Matrix Size (M=N=K)", fontsize=12)
    ax.set_ylabel("% of cuBLAS Performance", fontsize=12)
    ax.set_title("Kernel Performance Relative to cuBLAS", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_roofline(peak_tflops=19.5, peak_bw_tb=2.0, outfile="results/roofline.png"):
    """Roofline model for A100 with kernel data points."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Operational intensity range
    oi = np.logspace(-2, 3, 500)

    # Roofline: min(peak_compute, peak_bw * OI)
    # peak_compute in GFLOPS, peak_bw in GB/s
    peak_gflops = peak_tflops * 1000
    peak_bw = peak_bw_tb * 1000  # GB/s

    perf = np.minimum(peak_gflops, peak_bw * oi)

    ax.loglog(oi, perf, "k-", linewidth=2, label="Roofline")
    ax.axhline(y=peak_gflops, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=peak_gflops / peak_bw, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Operational Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=12)
    ax.set_title(f"Roofline Model — A100 ({peak_tflops} TFLOPS, {peak_bw_tb} TB/s)", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def parse_multi_gpu(filename):
    """Parse benchmark tables from bench_multi_gpu output."""
    sections = {
        "exp1": [],
        "exp2": [],
        "exp3": [],
        "exp4": [],
        "exp5": [],
        "exp6": [],
    }

    current = None
    with open(filename) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("===== Exp 1"):
                current = "exp1"
                continue
            if line.startswith("===== Exp 2"):
                current = "exp2"
                continue
            if line.startswith("===== Exp 3"):
                current = "exp3"
                continue
            if line.startswith("===== Exp 4"):
                current = "exp4"
                continue
            if line.startswith("===== Exp 5"):
                current = "exp5"
                continue
            if line.startswith("===== Exp 6"):
                current = "exp6"
                continue
            if (
                not current
                or not line
                or line.startswith("---")
                or line.startswith("M ")
                or line.startswith("Kernel")
                or line.startswith("Size")
            ):
                continue
            if line.startswith("Done"):
                break

            parts = line.split()
            try:
                if current == "exp1":
                    sections[current].append(
                        {
                            "M": int(parts[0]),
                            "N": int(parts[1]),
                            "K": int(parts[2]),
                            "gpus": int(parts[3]),
                            "gemm_ms": float(parts[4]),
                            "comm_ms": float(parts[5]),
                            "total_ms": float(parts[6]),
                            "gflops": float(parts[7]),
                        }
                    )
                elif current == "exp2":
                    sections[current].append(
                        {
                            "M": int(parts[0]),
                            "N_total": int(parts[1]),
                            "K": int(parts[2]),
                            "gpus": int(parts[3]),
                            "gemm_ms": float(parts[4]),
                            "comm_ms": float(parts[5]),
                            "total_ms": float(parts[6]),
                            "gflops": float(parts[7]),
                        }
                    )
                elif current == "exp3":
                    sections[current].append(
                        {
                            "size": int(parts[0]),
                            "gemm_ms": float(parts[1]),
                            "comm_ms": float(parts[2]),
                            "ratio": float(parts[3]),
                        }
                    )
                elif current == "exp4":
                    sections[current].append(
                        {
                            "kernel": parts[0],
                            "gemm_ms": float(parts[1]),
                            "comm_ms": float(parts[2]),
                            "ratio": float(parts[3]),
                        }
                    )
                elif current == "exp5":
                    sections[current].append(
                        {
                            "M": int(parts[0]),
                            "H": int(parts[1]),
                            "N": int(parts[2]),
                            "gpus": int(parts[3]),
                            "fwd_ms": float(parts[4]),
                            "bwd_ms": float(parts[5]),
                            "total_ms": float(parts[6]),
                        }
                    )
                elif current == "exp6":
                    sections[current].append(
                        {
                            "size": int(parts[0]),
                            "chunks": int(parts[1]),
                            "no_overlap_ms": float(parts[2]),
                            "overlap_ms": float(parts[3]),
                            "speedup": float(parts[4].rstrip("x")),
                        }
                    )
            except (IndexError, ValueError):
                continue

    return sections


def plot_strong_scaling(exp1, outfile="results/strong_scaling.png"):
    if not exp1:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    sizes = sorted(set(row["M"] for row in exp1))
    for size in sizes:
        rows = sorted([r for r in exp1 if r["M"] == size], key=lambda r: r["gpus"])
        ax.plot(
            [r["gpus"] for r in rows],
            [r["total_ms"] for r in rows],
            "o-",
            linewidth=2,
            label=f"Size {size}",
        )
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Strong Scaling — Column Parallel Forward")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_weak_scaling(exp2, outfile="results/weak_scaling.png"):
    if not exp2:
        return
    rows = sorted(exp2, key=lambda r: r["gpus"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r["gpus"] for r in rows], [r["total_ms"] for r in rows], "o-", linewidth=2)
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Weak Scaling — Fixed 2048x2048 Local Workload")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_ratio_vs_size(exp3, outfile="results/comm_compute_ratio_vs_size.png"):
    if not exp3:
        return
    rows = sorted(exp3, key=lambda r: r["size"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r["size"] for r in rows], [r["ratio"] for r in rows], "o-", linewidth=2)
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Comm / Compute Ratio")
    ax.set_title("Communication vs Compute Ratio by Matrix Size")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_ratio_vs_kernel(exp4, outfile="results/comm_compute_ratio_vs_kernel.png"):
    if not exp4:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    kernels = [r["kernel"] for r in exp4]
    ratios = [r["ratio"] for r in exp4]
    ax.bar(kernels, ratios, color=plt.cm.plasma(np.linspace(0.15, 0.85, len(exp4))))
    ax.set_ylabel("Comm / Compute Ratio")
    ax.set_title("Communication vs Compute Ratio by Local GEMM Kernel")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_mlp_timing(exp5, outfile="results/mlp_forward_backward.png"):
    if not exp5:
        return
    rows = sorted(exp5, key=lambda r: r["M"])
    sizes = [r["M"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, [r["fwd_ms"] for r in rows], "o-", linewidth=2, label="Forward")
    ax.plot(sizes, [r["bwd_ms"] for r in rows], "o-", linewidth=2, label="Backward")
    ax.plot(sizes, [r["total_ms"] for r in rows], "o-", linewidth=2, label="Total")
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Parallel MLP Forward/Backward Timing")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


def plot_overlap(exp6, outfile="results/overlap_speedup.png"):
    if not exp6:
        return
    rows = sorted(exp6, key=lambda r: r["size"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([r["size"] for r in rows], [r["speedup"] for r in rows], "o-", linewidth=2)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Matrix Size")
    ax.set_ylabel("Speedup")
    ax.set_title("Communication-Compute Overlap Speedup")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <single_gpu_results> [multi_gpu_results]")
        print("Generating roofline model only...")
        plot_roofline()
        sys.exit(0)

    sizes, kernels = parse_single_gpu(sys.argv[1])
    if sizes and kernels:
        plot_gflops_comparison(sizes, kernels)
        plot_cublas_percentage(sizes, kernels)
    plot_roofline()

    if len(sys.argv) >= 3:
        multi = parse_multi_gpu(sys.argv[2])
        plot_strong_scaling(multi["exp1"])
        plot_weak_scaling(multi["exp2"])
        plot_ratio_vs_size(multi["exp3"])
        plot_ratio_vs_kernel(multi["exp4"])
        plot_mlp_timing(multi["exp5"])
        plot_overlap(multi["exp6"])
