#!/bin/bash

# Configuration
ENGINE_PATH="../engines/gpt2-medium-with-past.plan"
CTX_LEN=20000
PAST_LEN=$((CTX_LEN - 1))
OUTPUT_FILE="gpt2_20k_corun_profile_$(date +%s)"

echo "Resetting and initializing NVIDIA MPS..."
pkill -9 nvidia-cuda-mps 2>/dev/null
sudo pkill -9 nvidia-cuda-mps 2>/dev/null
sleep 2

nvidia-cuda-mps-control -d
sleep 2

# Library Path Injection
VENV_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH="$VENV_PACKAGES/nvidia/cublas/lib:$VENV_PACKAGES/nvidia/cuda_runtime/lib:$VENV_PACKAGES/nvidia/cuda_nvrtc/lib:$VENV_PACKAGES/nvidia/curand/lib:$LD_LIBRARY_PATH"

# Fail-safe Cleanup (Only handles MPS now, workload.sh handles the Python kill)
cleanup() {
    echo "Shutting down MPS daemon to release GPU..."
    echo quit | nvidia-cuda-mps-control 2>/dev/null
    pkill -9 nvidia-cuda-mps 2>/dev/null
    sudo pkill -9 nvidia-cuda-mps 2>/dev/null
}
trap cleanup EXIT

# 1. Build Shapes
SHAPES="input_ids:1x1,position_ids:1x1,attention_mask:1x${CTX_LEN}"
for i in {0..23}; do
  SHAPES="${SHAPES},past_key_values.${i}.key:1x16x${PAST_LEN}x64,past_key_values.${i}.value:1x16x${PAST_LEN}x64"
done

echo "Initializing Nsys to trace the entire workload..."

# 2. Run Nsight Systems Profiler on the wrapper script
# We pass ENGINE_PATH and SHAPES as arguments to workload.sh
nsys profile \
    --trace=cuda,cudnn,cublas,nvtx,osrt \
    --sample=cpu \
    --stats=true \
    --output="../nsys/${OUTPUT_FILE}" \
    ./workload.sh "${ENGINE_PATH}" "${SHAPES}"

echo "Profiling complete. Results in ../nsys/${OUTPUT_FILE}.nsys-rep"