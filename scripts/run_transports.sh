#!/bin/bash
# ===========================================================================
# Cross-node benchmark transport sweep.
#
# Runs bench_multi_node once per transport configuration.  Every invocation
# uses a distinct MASTER_PORT (base + idx) so back-to-back runs don't trip
# over TIME_WAIT.  Same script runs on master (RANK=0) and every worker —
# because the loop body is identical in order, nodes stay in lockstep.
#
# Default sweep (override via TRANSPORTS env):
#   auto   — NCCL picks (IB if present, else TCP)
#   ib     — force InfiniBand
#   tcp    — force TCP socket (NCCL_IB_DISABLE=1)
#   ring   — IB + force ring algorithm
#
# NOTE: NCCL_ALGO=Tree is intentionally excluded -- NCCL's Tree algorithm
# does not support AllGather (our Exp 1-5 primary op), so the whole
# benchmark crashes at the first ncclAllGather call with
# "no algorithm/protocol available for function AllGather ... NCCL_ALGO
# was set to Tree".  Re-enable only for a ReduceScatter-only workload.
#
# Output files (master):
#   results/multi_node_${NTOTAL}gpu_<tag>.txt
# Worker logs:
#   results/multi_node_worker_rank${RANK}_<tag>.log
# ===========================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${RANK:?RANK not set}"

RESULTS_DIR="${RESULTS_DIR:-results}"
GPU_ARCH="${GPU_ARCH:-sm_90}"
BIN="./build/bench_multi_node"

mkdir -p "$RESULTS_DIR"

if [ ! -x "$BIN" ] || [ "${REBUILD:-0}" = "1" ]; then
    echo "Building bench_multi_node..."
    make bench_node GPU_ARCH="$GPU_ARCH"
fi

NLOCAL=$(nvidia-smi -L | wc -l)
NTOTAL=$((WORLD_SIZE * NLOCAL))
BASE_PORT="$MASTER_PORT"

# Each entry: "tag|KEY1=VAL1 KEY2=VAL2 ..."  (empty env string = NCCL defaults)
# Override by exporting TRANSPORTS as a newline-separated list before calling.
if [ -z "${TRANSPORTS:-}" ]; then
    TRANSPORTS='auto|
ib|NCCL_NET=IB NCCL_IB_DISABLE=0
tcp|NCCL_IB_DISABLE=1
ring|NCCL_ALGO=Ring'
fi

# Split TRANSPORTS into per-entry list (one per line)
mapfile -t ENTRIES <<< "$TRANSPORTS"

echo "=========================================="
echo "Transport sweep ($NTOTAL GPUs, rank $RANK)"
echo "Entries:"
printf '  %s\n' "${ENTRIES[@]}"
echo "=========================================="

run_one() {
    local tag="$1" envs="$2" port="$3"
    local out worker_log

    if [ "$RANK" = "0" ]; then
        out="$RESULTS_DIR/multi_node_${NTOTAL}gpu_${tag}.txt"
    else
        out="$RESULTS_DIR/multi_node_worker_rank${RANK}_${tag}.log"
    fi

    # Scrub previous NCCL transport env, then apply this entry's
    unset NCCL_NET NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_NET_GDR_LEVEL \
          NCCL_SOCKET_IFNAME NCCL_ALGO NCCL_PROTO NCCL_IB_HCA

    # Apply entry's env -- accept KEY=VAL tokens separated by spaces
    local kv
    for kv in $envs; do
        case "$kv" in
            *=*) export "$kv" ;;
        esac
    done

    # INFO is verbose but writes to stderr -- our parser reads stdout only, so
    # this doesn't corrupt the table.  It makes "unhandled cuda error" and
    # similar NCCL errors diagnosable after the fact.
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TRANSPORT_TAG="$tag"
    export MASTER_PORT="$port"

    echo ""
    echo "--- [$tag] MASTER_PORT=$port NCCL env: $envs ---"
    if [ "$RANK" = "0" ]; then
        "$BIN" 2>&1 | tee "$out"
    else
        "$BIN" > "$out" 2>&1
        echo "[rank $RANK] [$tag] done -> $out"
    fi
}

i=0
for entry in "${ENTRIES[@]}"; do
    [ -z "$entry" ] && continue
    tag="${entry%%|*}"
    envs="${entry#*|}"
    port=$((BASE_PORT + i))
    run_one "$tag" "$envs" "$port" || {
        echo "[rank $RANK] [$tag] FAILED (see $RESULTS_DIR for log); continuing sweep"
    }
    i=$((i + 1))
    # Small delay so master's next TCP listener has time to come up after
    # worker finishes its previous recv/close.
    sleep 2
done

echo ""
echo "Transport sweep finished on rank $RANK."
