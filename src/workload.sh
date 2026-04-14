#!/bin/bash

# $1 is ENGINE_PATH, $2 is SHAPES (passed from the main script)

echo "Starting compute-bound co-runner..."
python3 corunner.py &
CORUNNER_PID=$!

# Give CuPy time to JIT compile and warm up
sleep 4 

echo "Co-runner active. Starting TensorRT..."
trtexec \
    --loadEngine="$1" \
    --shapes="$2" \
    --iterations=3 \
    --duration=0 \
    --warmUp=1 \
    --streams=1 \
    --useManagedMemory=true \
    --profilingVerbosity=detailed

echo "TensorRT finished. Killing co-runner..."
kill $CORUNNER_PID 2>/dev/null
wait $CORUNNER_PID 2>/dev/null