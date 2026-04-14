#!/bin/bash

# Configuration
ENGINE_PATH="../engines/gpt2-medium-with-past.plan" # Replace with your actual engine path
CTX_LEN=20000
PAST_LEN=$((CTX_LEN - 1))
OUTPUT_FILE="gpt2_20k_profile"

# 1. Build the shapes string dynamically (GPT-2 Medium: 24 layers, 16 heads, 64 dim)
SHAPES="input_ids:1x1,position_ids:1x1,attention_mask:1x${CTX_LEN}"
for i in {0..23}; do
  SHAPES="${SHAPES},past_key_values.${i}.key:1x16x${PAST_LEN}x64,past_key_values.${i}.value:1x16x${PAST_LEN}x64"
done

echo "Running nsys profile for Context Length: $CTX_LEN..."

# 2. Run Nsight Systems Profiler
# Note: nsys generates a .nsys-rep file (the modern equivalent of .qdrep)
nsys profile \
    --trace=cuda,cudnn,cublas,nvtx,osrt \
    --sample=cpu \
    --stats=true \
    --force-overwrite=true \
    --output="../nsys/${OUTPUT_FILE}" \
    trtexec \
        --loadEngine="${ENGINE_PATH}" \
        --shapes="${SHAPES}" \
        --iterations=3 \
        --duration=0 \
        --warmUp=1 \
        --streams=1 \
        --useManagedMemory \
        --profilingVerbosity=detailed

echo "Profile saved to ${OUTPUT_FILE}.nsys-rep"