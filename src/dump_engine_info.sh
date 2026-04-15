#!/bin/bash

# Define the paths based on the repository structure
ENGINE_PATH="../engines/gpt2-medium-with-past.plan"
OUTPUT_JSON="layer_info.json"

echo "Dumping TensorRT engine layer execution graph to ${OUTPUT_JSON}..."

# Check if the engine exists
if [ ! -f "$ENGINE_PATH" ]; then
    echo "Warning: Engine file not found at $ENGINE_PATH"
    echo "Ensure that you have generated the engine before running this script."
    echo "Expected path: $ENGINE_PATH"
fi

# Run trtexec to dump the layer info
trtexec --loadEngine="${ENGINE_PATH}" --exportLayerInfo="${OUTPUT_JSON}"

if [ $? -eq 0 ]; then
    echo "Success! Layer info exported to ${OUTPUT_JSON}"
else
    echo "Error: trtexec failed to export layer info."
    echo "Make sure trtexec is installed and available in your PATH."
    exit 1
fi
