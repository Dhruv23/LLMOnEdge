#!/usr/bin/env python3
import os
import sys

try:
    import onnx
    from onnx import shape_inference
except ImportError:
    print("Error: The 'onnx' library is required to run this script.", file=sys.stderr)
    print("Please install it using: pip install onnx", file=sys.stderr)
    sys.exit(1)

def get_tensor_shapes(graph):
    """Returns a dictionary mapping tensor names to their shapes."""
    shapes = {}

    # Process inputs
    for tensor in graph.input:
        shape = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        shapes[tensor.name] = f"[{','.join(shape)}]"

    # Process outputs
    for tensor in graph.output:
        shape = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        shapes[tensor.name] = f"[{','.join(shape)}]"

    # Process value_info (intermediate tensors after shape inference)
    for tensor in graph.value_info:
        shape = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        shapes[tensor.name] = f"[{','.join(shape)}]"

    return shapes

def inspect_onnx_graph(onnx_path):
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at '{onnx_path}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading ONNX model from {onnx_path}...")
    model = onnx.load(onnx_path)

    print("Running shape inference...")
    # Infer shapes so we can print intermediate tensor dimensions
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Shape inference failed: {e}")
        print("Continuing with available shape information.")

    graph = model.graph

    print("Extracting tensor shapes...")
    tensor_shapes = get_tensor_shapes(graph)

    print("\n" + "="*80)
    print("Sequential Flow of Nodes:")
    print("="*80)

    for i, node in enumerate(graph.node):
        print(f"Node {i}:")
        print(f"  Name:    {node.name if node.name else '<unnamed>'}")
        print(f"  OpType:  {node.op_type}")

        # Inputs
        print("  Inputs:")
        for input_name in node.input:
            if input_name:
                shape = tensor_shapes.get(input_name, "[unknown shape]")
                print(f"    - {input_name}: {shape}")

        # Outputs
        print("  Outputs:")
        for output_name in node.output:
            if output_name:
                shape = tensor_shapes.get(output_name, "[unknown shape]")
                print(f"    - {output_name}: {shape}")

        print("-" * 40)

if __name__ == "__main__":
    default_path = "../onnx/gpt2-medium-with-past/model.onnx"
    onnx_file = sys.argv[1] if len(sys.argv) > 1 else default_path
    inspect_onnx_graph(onnx_file)
