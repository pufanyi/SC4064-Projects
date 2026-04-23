#!/usr/bin/env python3
"""Run benchmarks and collect results into a single JSON file.

Modes
-----
Run locally and collect (legacy behaviour, single-node only):
    python scripts/run_and_collect.py [NUM_GPUS]

Re-parse existing JSON raw fields:
    python scripts/run_and_collect.py --reparse

Merge already-collected text files (works for multi-node + transport sweep):
    python scripts/run_and_collect.py --merge \
        --single-gpu   results/single_gpu.txt \
        --single-node  results/multi_gpu.txt \
        --multi-node   results/multi_node_16gpu.txt
        # or, with a transport sweep:
        --multi-node   auto=results/multi_node_16gpu_auto.txt \
        --multi-node   ib=results/multi_node_16gpu_ib.txt \
        --multi-node   tcp=results/multi_node_16gpu_tcp.txt

The merged JSON has these sections:
    single_gpu              -- kernel correctness + GFLOPS table
    multi_gpu               -- single-node scaling (threaded, up to 8 GPUs)
    multi_node              -- cross-node metadata + primary transport data
    multi_node.transports   -- keyed-by-tag per-transport tables (if sweep)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
RESULTS = ROOT / "results"


# ── subprocess helper ────────────────────────────────────────────────────


def run(cmd, **kw):
    print(f">>> {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(f"Command failed: {' '.join(map(str, cmd))}")
    return r.stdout


# ── single-GPU text parser ───────────────────────────────────────────────


def parse_single_gpu(text):
    result = {"correctness": {}, "performance": {}, "stddev_pct": {}, "detail": {}}

    for m in re.finditer(r"^(\S+)\s*:\s*(PASS|FAIL).*", text, re.MULTILINE):
        result["correctness"][m.group(1)] = m.group(2)

    # Loose-match title so the parser still works whether the binary prints
    # "Performance Benchmark (GFLOPS)" or "Performance Benchmark (GFLOPS, mean of N)".
    m = re.search(r"Performance Benchmark \(GFLOPS[^)]*\)", text)
    if not m:
        return result
    block = text[m.end() :]

    # Stop at the next section header so we don't bleed into the stddev table.
    nxt = re.search(r"^=====", block, re.MULTILINE)
    perf_block = block[: nxt.start()] if nxt else block

    def _parse_kernel_by_size(segment: str):
        lines = [ln for ln in segment.strip().splitlines() if ln.strip()]
        header_line = None
        data_lines = []
        for ln in lines:
            s = ln.strip()
            if not s or set(s).issubset({"-", " ", "="}):
                continue
            if header_line is None:
                header_line = ln
            else:
                data_lines.append(ln)
        if not header_line:
            return {}, []
        sizes = []
        for p in header_line.split():
            with contextlib.suppress(ValueError):
                sizes.append(int(p))
        if not sizes:
            return {}, []
        rows = {}
        for ln in data_lines:
            if ln.strip().startswith("Done") or ln.strip().startswith("====="):
                break
            parts = ln.split()
            if len(parts) < 2:
                continue
            kernel = parts[0]
            try:
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            if len(vals) != len(sizes):
                continue
            rows[kernel] = {str(s): v for s, v in zip(sizes, vals, strict=True)}
        return rows, sizes

    perf_rows, _ = _parse_kernel_by_size(perf_block)
    result["performance"] = perf_rows

    # Optional: stddev table (CoV %)
    m_std = re.search(r"Performance Stddev \([^)]*\)", text)
    if m_std:
        std_block = text[m_std.end() :]
        nxt = re.search(r"^=====", std_block, re.MULTILINE)
        std_block = std_block[: nxt.start()] if nxt else std_block
        std_rows, _ = _parse_kernel_by_size(std_block)
        result["stddev_pct"] = std_rows

    # Optional: detailed timing table (size is whatever the binary picked).
    m_det = re.search(r"Detailed Timing at M=N=K=(\d+)[^\n]*\n", text)
    if m_det:
        result["detail_size"] = int(m_det.group(1))
        det_block = text[m_det.end() :]
        nxt = re.search(r"^=====|^\nDone\.", det_block, re.MULTILINE)
        det_block = det_block[: nxt.start()] if nxt else det_block
        for line in det_block.splitlines():
            s = line.strip()
            if not s or set(s).issubset({"-", " ", "="}) or s.startswith("Kernel"):
                continue
            if s.startswith("Done"):
                break
            parts = line.split()
            if len(parts) != 6:
                continue
            k = parts[0]
            try:
                mean, median, stddev, mn, mx = (float(x) for x in parts[1:])
            except ValueError:
                continue
            result["detail"][k] = {
                "mean": mean,
                "median": median,
                "stddev": stddev,
                "min": mn,
                "max": mx,
            }

    return result


# ── multi-GPU / multi-node text parser (same output format) ──────────────


def parse_experiments(text):
    """Parse the '===== Exp N: title =====' blocks emitted by both
    bench_multi_gpu and bench_multi_node.  Returns {'exp1': {...}, ...}."""
    result = {}

    exp_re = re.compile(
        r"={3,}\s*Exp\s+(\d+):\s*(.*?)\s*={3,}\n(.*?)(?=\n={3,}\s*Exp|\nDone\.|$)",
        re.DOTALL,
    )

    for m in exp_re.finditer(text):
        exp_num = int(m.group(1))
        exp_title = m.group(2).strip()
        body = m.group(3).strip()

        lines = [line for line in body.splitlines() if line.strip()]
        header = None
        data = []
        for line in lines:
            stripped = line.strip()
            if not stripped or set(stripped).issubset({"-", " ", "="}):
                continue
            if header is None:
                header = line
            else:
                data.append(line)

        if not header or not data:
            continue

        cols = header.split()
        rows = []
        for line in data:
            vals = line.split()
            if len(vals) != len(cols):
                continue
            # Reject rows that aren't benchmark data -- the first column is
            # always an integer M (or a kernel name in Exp 4).  Interleaved
            # NCCL INFO / WARN lines occasionally match len(cols) by accident
            # (e.g. "[Proxy Progress] Device 1 CPU core 22"), so drop anything
            # whose first token isn't an int and isn't a known kernel name.
            first = vals[0]
            _KERNELS = {
                "Naive",
                "Coalesced",
                "Uncoalesced",
                "SmemTiling",
                "BlockTile1D",
                "BlockTile2D",
                "Vectorized",
                "WarpTile",
                "cuBLAS",
            }
            try:
                int(first)
            except ValueError:
                if first not in _KERNELS:
                    continue
            row = {}
            for c, v in zip(cols, vals, strict=True):
                c = re.sub(r"\(.*?\)$", "", c)
                try:
                    if "." in v or "x" in v:
                        row[c] = float(v.rstrip("x"))
                    else:
                        row[c] = int(v)
                except ValueError:
                    row[c] = v
            rows.append(row)

        result[f"exp{exp_num}"] = {"title": exp_title, "data": rows}

    return result


# Back-compat alias (plot_analysis.py / older callers may import this name)
parse_multi_gpu = parse_experiments


# ── metadata ─────────────────────────────────────────────────────────────


def _nvidia_smi_first_gpu():
    try:
        out = run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"]
        )
        return out.strip().splitlines()[0].strip()
    except SystemExit:
        return "unknown"


# ── modes ────────────────────────────────────────────────────────────────


def mode_run_and_collect(max_gpus: int, outpath: Path):
    """Legacy behaviour: run the binaries in the current process."""
    out = {}

    print("=== Running single-GPU benchmark ===", flush=True)
    t0 = time.time()
    raw_single = run([str(BUILD / "bench_single_gpu")])
    t1 = time.time()
    out["single_gpu"] = parse_single_gpu(raw_single)
    out["single_gpu"]["raw"] = raw_single
    out["single_gpu"]["elapsed_s"] = round(t1 - t0, 2)
    print(f"    done in {t1 - t0:.1f}s", flush=True)

    print(f"=== Running multi-GPU benchmark ({max_gpus} GPUs) ===", flush=True)
    t0 = time.time()
    raw_multi = run([str(BUILD / "bench_multi_gpu"), str(max_gpus)])
    t1 = time.time()
    out["multi_gpu"] = parse_experiments(raw_multi)
    out["multi_gpu"]["raw"] = raw_multi
    out["multi_gpu"]["elapsed_s"] = round(t1 - t0, 2)
    out["multi_gpu"]["max_gpus"] = max_gpus
    out["multi_gpu"]["kernel"] = "cuBLAS"
    print(f"    done in {t1 - t0:.1f}s", flush=True)

    out["metadata"] = {
        "gpu": _nvidia_smi_first_gpu(),
        "num_gpus": max_gpus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    outpath.parent.mkdir(exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nResults written to {outpath}")


def mode_reparse(outpath: Path):
    with outpath.open() as f:
        out = json.load(f)
    if "single_gpu" in out and "raw" in out["single_gpu"]:
        raw = out["single_gpu"]["raw"]
        out["single_gpu"] = {
            **parse_single_gpu(raw),
            "raw": raw,
            **{k: out["single_gpu"][k] for k in out["single_gpu"] if k == "elapsed_s"},
        }
    if "multi_gpu" in out and "raw" in out["multi_gpu"]:
        raw = out["multi_gpu"]["raw"]
        keep = {
            k: out["multi_gpu"][k]
            for k in out["multi_gpu"]
            if k in ("elapsed_s", "max_gpus", "kernel")
        }
        out["multi_gpu"] = {**parse_experiments(raw), "raw": raw, **keep}
    if "multi_node" in out:
        mn = out["multi_node"]
        # Reparse each transport variant if transports.{tag}.raw exists
        if "transports" in mn:
            for tag, tdata in mn["transports"].items():
                if "raw" in tdata:
                    raw = tdata["raw"]
                    keep = {k: tdata[k] for k in tdata if k in ("transport_tag", "source_file")}
                    mn["transports"][tag] = {**parse_experiments(raw), "raw": raw, **keep}
            # Refresh the promoted top-level primary transport view
            primary = mn.get("primary_transport") or next(iter(mn["transports"]))
            for k in ("exp1", "exp2", "exp3", "exp4", "exp5", "exp6"):
                if k in mn["transports"][primary]:
                    mn[k] = mn["transports"][primary][k]
            mn["raw"] = mn["transports"][primary].get("raw", mn.get("raw", ""))
        elif "raw" in mn:
            raw = mn["raw"]
            keep = {
                k: mn[k]
                for k in mn
                if k
                in (
                    "elapsed_s",
                    "kernel",
                    "num_nodes",
                    "gpus_per_node",
                    "world_size",
                    "primary_transport",
                )
            }
            out["multi_node"] = {**parse_experiments(raw), "raw": raw, **keep}
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Re-parsed and written to {outpath}")


def _extract_world_size(text: str) -> int | None:
    m = re.search(r"World size:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _extract_nodes_ppn(text: str) -> tuple[int | None, int | None]:
    nodes = re.search(r"Nodes:\s*(\d+)", text)
    ppn = re.search(r"GPUs per node:\s*(\d+)", text)
    return (int(nodes.group(1)) if nodes else None, int(ppn.group(1)) if ppn else None)


def _extract_transport_tag(text: str) -> str | None:
    m = re.search(r"Transport tag:\s*(\S+)", text)
    return m.group(1) if m else None


def _parse_multi_node_spec(spec: str) -> tuple[str | None, Path]:
    """Accept 'tag=path' or bare 'path'.  When bare, tag is inferred from
    filename pattern multi_node_<N>gpu_<tag>.txt, else None."""
    if "=" in spec and not Path(spec).exists():
        tag, _, path = spec.partition("=")
        return tag.strip(), Path(path.strip())
    path = Path(spec)
    m = re.search(r"multi_node_\d+gpu_(.+)\.txt$", path.name)
    return (m.group(1) if m else None, path)


def _build_multi_node_section(paths: list[str]) -> dict | None:
    """Parse one or more multi-node text files into a single section."""
    if not paths:
        return None

    entries = [_parse_multi_node_spec(s) for s in paths]
    # Filter to existing files
    entries = [(tag, p) for tag, p in entries if p.exists()]
    if not entries:
        return None

    section: dict = {}
    transports: dict = {}
    for tag, p in entries:
        raw = p.read_text()
        parsed = parse_experiments(raw)
        tag = tag or _extract_transport_tag(raw) or "default"
        transports[tag] = {
            **parsed,
            "raw": raw,
            "transport_tag": _extract_transport_tag(raw),
            "source_file": str(p),
        }

    # Metadata comes from the first available file
    raw0 = entries[0][1].read_text()
    ws = _extract_world_size(raw0)
    nodes, ppn = _extract_nodes_ppn(raw0)
    section["world_size"] = ws
    section["num_nodes"] = nodes
    section["gpus_per_node"] = ppn
    section["kernel"] = "cuBLAS"
    section["transports"] = transports

    # Promote one primary transport to the top level for backward-compatible
    # plotting.  Priority: 'auto' > 'default' > first available.
    primary = None
    for pref in ("auto", "default"):
        if pref in transports:
            primary = pref
            break
    if primary is None:
        primary = next(iter(transports))
    section["primary_transport"] = primary
    for k in ("exp1", "exp2", "exp3", "exp4", "exp5", "exp6"):
        if k in transports[primary]:
            section[k] = transports[primary][k]
    section["raw"] = transports[primary].get("raw", "")

    return section


def mode_merge(
    single_gpu: Path | None, single_node: Path | None, multi_node: list[str], outpath: Path
):
    """Merge pre-collected text files into JSON.  Missing files are skipped;
    existing sections in outpath are preserved if the corresponding text
    file is not provided.  multi_node accepts multiple transport variants."""
    out = {}
    if outpath.exists():
        with outpath.open() as f:
            out = json.load(f)

    if single_gpu and single_gpu.exists():
        raw = single_gpu.read_text()
        out["single_gpu"] = {**parse_single_gpu(raw), "raw": raw}

    if single_node and single_node.exists():
        raw = single_node.read_text()
        ngpus = re.search(r"Max GPUs used:\s*(\d+)", raw)
        out["multi_gpu"] = {
            **parse_experiments(raw),
            "raw": raw,
            "max_gpus": int(ngpus.group(1)) if ngpus else None,
            "kernel": "cuBLAS",
        }

    mn = _build_multi_node_section(multi_node or [])
    if mn is not None:
        out["multi_node"] = mn

    out.setdefault("metadata", {})
    out["metadata"].update(
        {
            "gpu": _nvidia_smi_first_gpu(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )

    outpath.parent.mkdir(exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Merged -> {outpath}")

    secs = [k for k in ("single_gpu", "multi_gpu", "multi_node") if k in out]
    print(f"Sections present: {', '.join(secs)}")
    if "multi_node" in out and "transports" in out["multi_node"]:
        tags = list(out["multi_node"]["transports"].keys())
        primary = out["multi_node"].get("primary_transport")
        print(f"Transports: {', '.join(tags)}  (primary={primary})")


# ── entry point ──────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--reparse", action="store_true", help="reparse raw fields in existing JSON")
    p.add_argument("--merge", action="store_true", help="merge text files into JSON")
    p.add_argument("--single-gpu", type=Path, help="single-GPU text output")
    p.add_argument("--single-node", type=Path, help="single-node multi-GPU text output")
    p.add_argument(
        "--multi-node",
        action="append",
        default=[],
        help="multi-node text output; repeat as 'tag=path' or plain "
        "'path' (tag inferred from multi_node_NNgpu_<tag>.txt)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=RESULTS / "benchmark_results.json",
        help="JSON output path (default: results/benchmark_results.json)",
    )
    p.add_argument(
        "num_gpus",
        nargs="?",
        type=int,
        default=8,
        help="max GPUs for legacy run-and-collect mode (default 8)",
    )
    args = p.parse_args()

    if args.reparse:
        mode_reparse(args.out)
    elif args.merge:
        mode_merge(args.single_gpu, args.single_node, args.multi_node, args.out)
    else:
        mode_run_and_collect(args.num_gpus, args.out)


if __name__ == "__main__":
    main()
