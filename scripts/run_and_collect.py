#!/usr/bin/env python3
"""Run single-GPU and multi-GPU benchmarks and collect results into JSON."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def run(cmd, **kw):
    print(f">>> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(f"Command failed: {' '.join(cmd)}")
    return r.stdout


# ── single-GPU ────────────────────────────────────────────────────────────

def parse_single_gpu(text):
    result = {"correctness": {}, "performance": {}}

    # Correctness: "kernel_name      : PASS ..." or "FAIL"
    for m in re.finditer(r"^(\S+)\s*:\s*(PASS|FAIL).*", text, re.M):
        result["correctness"][m.group(1)] = m.group(2)

    # Performance table
    perf_block = text.split("Performance Benchmark (GFLOPS)")
    if len(perf_block) < 2:
        return result
    block = perf_block[1]
    lines = [l for l in block.strip().splitlines() if l.strip()]
    # First non-dash line is header
    header_line = None
    data_lines = []
    for l in lines:
        if set(l.strip()) == {"-"}:
            continue
        if header_line is None:
            header_line = l
        else:
            data_lines.append(l)

    if header_line is None:
        return result

    # Parse sizes from header
    parts = header_line.split()
    sizes = [int(x) for x in parts[1:]]  # skip "Kernel"

    for line in data_lines:
        if line.strip().startswith("Done"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        kernel = parts[0]
        gflops = [float(x) for x in parts[1:]]
        result["performance"][kernel] = {
            str(s): g for s, g in zip(sizes, gflops)
        }

    return result


# ── multi-GPU ─────────────────────────────────────────────────────────────

def parse_multi_gpu(text):
    result = {}

    # Split by "===== Exp" headers
    exp_re = re.compile(
        r"={3,}\s*Exp\s+(\d+):\s*(.*?)\s*={3,}\n(.*?)(?=\n={3,}\s*Exp|\nDone\.|$)",
        re.S,
    )

    for m in exp_re.finditer(text):
        exp_num = int(m.group(1))
        exp_title = m.group(2).strip()
        body = m.group(3).strip()

        lines = [l for l in body.splitlines() if l.strip()]
        # Find header line (first non-dash line)
        header = None
        data = []
        for l in lines:
            if set(l.strip()).issubset({"-", " "}):
                continue
            if header is None:
                header = l
            else:
                data.append(l)

        if not header or not data:
            continue

        cols = header.split()
        rows = []
        for line in data:
            vals = line.split()
            if len(vals) != len(cols):
                continue
            row = {}
            for c, v in zip(cols, vals):
                c = c.rstrip("(ms)").rstrip("(")  # clean column names
                try:
                    # try int first, then float
                    if "." in v or "x" in v:
                        row[c] = float(v.rstrip("x"))
                    else:
                        row[c] = int(v)
                except ValueError:
                    row[c] = v
            rows.append(row)

        result[f"exp{exp_num}"] = {"title": exp_title, "data": rows}

    return result


# ── main ──────────────────────────────────────────────────────────────────

def main():
    max_gpus = 8

    out = {}

    # Single-GPU
    print("=== Running single-GPU benchmark ===", flush=True)
    t0 = time.time()
    raw_single = run([str(BUILD / "bench_single_gpu")])
    t1 = time.time()
    out["single_gpu"] = parse_single_gpu(raw_single)
    out["single_gpu"]["raw"] = raw_single
    out["single_gpu"]["elapsed_s"] = round(t1 - t0, 2)
    print(f"    done in {t1-t0:.1f}s", flush=True)

    # Multi-GPU (cuBLAS kernel = 7, which is the default)
    print(f"=== Running multi-GPU benchmark ({max_gpus} GPUs) ===", flush=True)
    t0 = time.time()
    raw_multi = run([str(BUILD / "bench_multi_gpu"), str(max_gpus)])
    t1 = time.time()
    out["multi_gpu"] = parse_multi_gpu(raw_multi)
    out["multi_gpu"]["raw"] = raw_multi
    out["multi_gpu"]["elapsed_s"] = round(t1 - t0, 2)
    out["multi_gpu"]["max_gpus"] = max_gpus
    out["multi_gpu"]["kernel"] = "cuBLAS"
    print(f"    done in {t1-t0:.1f}s", flush=True)

    # Metadata
    nvsmi = run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
                  "--format=csv,noheader"])
    out["metadata"] = {
        "gpu": nvsmi.strip().splitlines()[0].strip(),
        "num_gpus": max_gpus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    outpath = ROOT / "results" / "benchmark_results.json"
    outpath.parent.mkdir(exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {outpath}")


if __name__ == "__main__":
    main()
