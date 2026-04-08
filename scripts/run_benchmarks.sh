#!/bin/bash
# ===========================================================================
# Run all benchmarks and save results
# ===========================================================================
set -e

RESULTS_DIR="results"
mkdir -p $RESULTS_DIR

echo "=========================================="
echo " Building..."
echo "=========================================="
make clean
make all

echo ""
echo "=========================================="
echo " Single-GPU Benchmark"
echo "=========================================="
./build/bench_single_gpu 2>&1 | tee $RESULTS_DIR/single_gpu.txt

echo ""
echo "=========================================="
echo " Multi-GPU Benchmark"
echo "=========================================="
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -ge 2 ]; then
    ./build/bench_multi_gpu $NUM_GPUS 2>&1 | tee $RESULTS_DIR/multi_gpu.txt
else
    echo "Only 1 GPU available — skipping multi-GPU benchmark."
    echo "Run on a multi-GPU node for tensor parallelism experiments."
fi

echo ""
echo "=========================================="
echo " Generating Plots"
echo "=========================================="
python3 scripts/plot_results.py $RESULTS_DIR/single_gpu.txt $RESULTS_DIR/multi_gpu.txt

echo ""
echo "All results saved to $RESULTS_DIR/"
