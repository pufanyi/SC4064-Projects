#!/bin/bash
# ===========================================================================
# One-click driver: build, run every experiment, merge results into JSON.
#
# Execution order is chosen so master and worker spend as little time idling
# on each other as possible:
#
#   Stage 1  Build                                 (both roles)
#   Stage 2  Cross-node transport sweep            (both roles, rendezvous)
#   Stage 3  Single-GPU bench                      (master only)
#   Stage 4  Single-node multi-GPU bench           (master only)
#   Stage 5  Merge JSON                            (master only)
#   Stage 6  Plots                                 (master only)
#
# Worker exits after stage 2; master continues with the local stages.
#
# Same script runs on every pod of a PyTorchJob.  Env vars expected
# (injected by the K8s orchestrator):
#   MASTER_ADDR, MASTER_PORT, WORLD_SIZE (NODES, not GPUs), RANK
#
# Override knobs:
#   GPU_ARCH         (default sm_90 = H100; use sm_80 for A100)
#   SKIP_BUILD, SKIP_SINGLE_GPU, SKIP_SINGLE_NODE, SKIP_MULTI_NODE, SKIP_PLOTS
# ===========================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# -- Config ------------------------------------------------------------------
GPU_ARCH="${GPU_ARCH:-sm_90}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_SINGLE_GPU="${SKIP_SINGLE_GPU:-0}"
SKIP_SINGLE_NODE="${SKIP_SINGLE_NODE:-0}"
SKIP_MULTI_NODE="${SKIP_MULTI_NODE:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"

RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
IS_MASTER=1
[ "$RANK" != "0" ] && IS_MASTER=0

NLOCAL=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
NTOTAL=$((WORLD_SIZE * NLOCAL))

mkdir -p "$RESULTS_DIR"

hrule() { printf '%.0s=' {1..70}; printf '\n'; }
banner() { echo ""; hrule; echo " $*"; hrule; }

SINGLE_GPU_OUT="$RESULTS_DIR/single_gpu.txt"
SINGLE_NODE_OUT="$RESULTS_DIR/multi_gpu.txt"

# NCCL_DEBUG=INFO

# -- 1. Build (both roles) ---------------------------------------------------
if [ "$SKIP_BUILD" != "1" ]; then
    banner "Stage 1/6: Build (GPU_ARCH=$GPU_ARCH, rank $RANK)"
    make all GPU_ARCH="$GPU_ARCH"
else
    echo "info: SKIP_BUILD=1 -- skipping build"
fi

# -- 2. Cross-node transport sweep (both roles) ------------------------------
# Run this FIRST so master and worker meet at the NCCL rendezvous quickly
# after build.  Otherwise worker would sit idle through the long single-node
# stages.  Worker exits after this stage; master continues below.

if [ "$SKIP_MULTI_NODE" != "1" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    banner "Stage 2/6: Cross-node benchmark (rank $RANK of $WORLD_SIZE, ${NTOTAL} GPUs total)"
    : "${MASTER_ADDR:?MASTER_ADDR not set (required when WORLD_SIZE>1)}"
    : "${MASTER_PORT:?MASTER_PORT not set (required when WORLD_SIZE>1)}"

    if [ "${SKIP_MULTI_NODE_SWEEP:-0}" = "1" ]; then
        tag="${MULTI_NODE_SINGLE_TAG:-default}"
        export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
        export NCCL_ASYNC_ERROR_HANDLING=1
        export TRANSPORT_TAG="$tag"
        out="$RESULTS_DIR/multi_node_${NTOTAL}gpu_${tag}.txt"
        if [ "$IS_MASTER" = "1" ]; then
            ./build/bench_multi_node 2>&1 | tee "$out"
        else
            ./build/bench_multi_node > "$RESULTS_DIR/multi_node_worker_rank${RANK}_${tag}.log" 2>&1
            echo "[rank $RANK] finished"
        fi
    else
        RESULTS_DIR="$RESULTS_DIR" GPU_ARCH="$GPU_ARCH" bash scripts/run_transports.sh
    fi
elif [ "$WORLD_SIZE" -le 1 ]; then
    echo "info: WORLD_SIZE=$WORLD_SIZE (single-node), skipping cross-node stage"
fi

# -- Worker exits here (no more stages for worker) ---------------------------
if [ "$IS_MASTER" != "1" ]; then
    banner "Worker done (rank $RANK)"
    exit 0
fi

# -- 3. Single-GPU bench (master only) ---------------------------------------
if [ "$SKIP_SINGLE_GPU" != "1" ]; then
    banner "Stage 3/6: Single-GPU benchmark"
    ./build/bench_single_gpu 2>&1 | tee "$SINGLE_GPU_OUT"
fi

# -- 4. Single-node multi-GPU bench (master only) ----------------------------
if [ "$SKIP_SINGLE_NODE" != "1" ] && [ "$NLOCAL" -ge 2 ]; then
    banner "Stage 4/6: Single-node multi-GPU benchmark ($NLOCAL GPUs)"
    ./build/bench_multi_gpu "$NLOCAL" 2>&1 | tee "$SINGLE_NODE_OUT"
elif [ "$SKIP_SINGLE_NODE" != "1" ]; then
    echo "info: only $NLOCAL GPU visible -- skipping single-node multi-GPU stage"
fi

# -- 5. Merge JSON -----------------------------------------------------------
banner "Stage 5/6: Merge -> $RESULTS_DIR/benchmark_results.json"

MERGE_ARGS=()
[ -f "$SINGLE_GPU_OUT" ]   && MERGE_ARGS+=(--single-gpu   "$SINGLE_GPU_OUT")
[ -f "$SINGLE_NODE_OUT" ]  && MERGE_ARGS+=(--single-node  "$SINGLE_NODE_OUT")

shopt -s nullglob
for f in "$RESULTS_DIR"/multi_node_${NTOTAL}gpu_*.txt \
         "$RESULTS_DIR"/multi_node_${NTOTAL}gpu.txt; do
    [ -f "$f" ] && MERGE_ARGS+=(--multi-node "$f")
done
shopt -u nullglob

if [ ${#MERGE_ARGS[@]} -gt 0 ]; then
    uv run python scripts/run_and_collect.py --merge "${MERGE_ARGS[@]}"
else
    echo "info: no text outputs to merge"
fi

# -- 6. Plots ----------------------------------------------------------------
if [ "$SKIP_PLOTS" != "1" ]; then
    banner "Stage 6/6: Plots"
    if [ -f scripts/plot_analysis.py ]; then
        uv run python scripts/plot_analysis.py || echo "warning: plot_analysis.py failed"
    fi
    if [ -f scripts/plot_results.py ] && [ -f "$SINGLE_GPU_OUT" ] && [ -f "$SINGLE_NODE_OUT" ]; then
        uv run python scripts/plot_results.py "$SINGLE_GPU_OUT" "$SINGLE_NODE_OUT" || true
    fi
fi

banner "Done. Outputs in $RESULTS_DIR/"
ls -lh "$RESULTS_DIR"/*.txt "$RESULTS_DIR"/*.json 2>/dev/null || true
