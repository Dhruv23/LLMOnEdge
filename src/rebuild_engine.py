#!/usr/bin/env python3
import subprocess
import os

# GPT-2 Medium Specs
N_LAYERS = 24
N_HEADS = 16
HDIM = 64

# Target Context Lengths - MODIFIED FOR 20K BUILD SUCCESS
MIN_CTX = 1      
OPT_CTX = 128    # <--- LOWERED from 1024 to save build-time memory
MAX_CTX = 30000  

def build_shapes():
    min_shapes = [f"input_ids:1x1", f"position_ids:1x1", f"attention_mask:1x{MIN_CTX}"]
    opt_shapes = [f"input_ids:1x1", f"position_ids:1x1", f"attention_mask:1x{OPT_CTX}"]
    max_shapes = [f"input_ids:1x1", f"position_ids:1x1", f"attention_mask:1x{MAX_CTX}"]

    min_past = max(0, MIN_CTX - 1)
    opt_past = max(0, OPT_CTX - 1)
    max_past = max(0, MAX_CTX - 1)

    for i in range(N_LAYERS):
        min_shapes.append(f"past_key_values.{i}.key:1x{N_HEADS}x{min_past}x{HDIM}")
        opt_shapes.append(f"past_key_values.{i}.key:1x{N_HEADS}x{opt_past}x{HDIM}")
        max_shapes.append(f"past_key_values.{i}.key:1x{N_HEADS}x{max_past}x{HDIM}")
        min_shapes.append(f"past_key_values.{i}.value:1x{N_HEADS}x{min_past}x{HDIM}")
        opt_shapes.append(f"past_key_values.{i}.value:1x{N_HEADS}x{opt_past}x{HDIM}")
        max_shapes.append(f"past_key_values.{i}.value:1x{N_HEADS}x{max_past}x{HDIM}")

    return ",".join(min_shapes), ",".join(opt_shapes), ",".join(max_shapes)

# ... (Keep your build_shapes function as is) ...

def main():
    onnx_path = "../onnx/gpt2-medium-with-past/model.onnx"
    engine_path = "../engines/gpt2-medium-with-past.plan"

    if os.path.exists(engine_path):
        os.remove(engine_path)

    min_sh, opt_sh, max_sh = build_shapes()

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_sh}",
        f"--optShapes={opt_sh}",
        f"--maxShapes={max_sh}",
        "--fp16",
        "--memPoolSize=workspace:512",      # Aggressively lowered
        "--avgTiming=1",                    # Minimize timing overhead
        "--preview=+runtimeActivationResize",
        "--stronglyTyped"                   # Newer TRT optimization path
    ]

    print(f"🚀 Attempting frugal recompile for 20k...")
    subprocess.run(cmd, check=True)
    print(f"\n✅ Engine rebuilt: {engine_path}")
if __name__ == "__main__":
    main()