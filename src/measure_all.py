#!/usr/bin/env python3
"""
Step 6b: ML Dataset Generator
Runs trtexec and exports EVERY single inference latency to a CSV.
Designed for training ML regressors (Random Forest, XGBoost, etc).

Output Format (dataset.csv):
timestamp, engine, ctx_len, run_id, inference_idx, latency_ms, compute_ms
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

# GPT-2 Medium constants
N_LAYERS = 24
N_HEADS = 16
HDIM = 64

def load_plan(plan_path: str) -> dict:
    with open(plan_path, "r") as f:
        return json.load(f)

def build_shapes_str(ctx_len: int) -> str:
    past_len = max(0, ctx_len - 1)
    parts = []
    parts.append(f"input_ids:1x1")
    parts.append(f"position_ids:1x1")
    parts.append(f"attention_mask:1x{ctx_len}")
    for i in range(N_LAYERS):
        parts.append(f"past_key_values.{i}.key:1x{N_HEADS}x{past_len}x{HDIM}")
        parts.append(f"past_key_values.{i}.value:1x{N_HEADS}x{past_len}x{HDIM}")
    return ",".join(parts)

def run_trtexec_and_export(engine_path: str, shapes: str, iters: int, warmup_ms: int, 
                           no_data_transfers: bool, json_path: str):
    """
    Runs trtexec and forces it to dump raw timings to json_path.
    """
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes={shapes}",
        f"--iterations={iters}",
        "--duration=0",
        f"--warmUp={warmup_ms}",
        "--streams=1",
        "--verbose=0",
        f"--exportTimes={json_path}"  # <--- The magic flag
    ]

    if no_data_transfers:
        cmd.append("--noDataTransfers")

    # Run silently
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed:\n{proc.stdout}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default="training_dataset.csv", help="Output CSV for ML training")
    ap.add_argument("--iters", type=int, default=100, help="Datapoints per bucket")
    ap.add_argument("--runs", type=int, default=1, help="Repeated runs per bucket")
    ap.add_argument("--warmup_ms", type=int, default=200)
    ap.add_argument("--no_data_transfers", action="store_true")
    
    args = ap.parse_args()

    plan = load_plan(args.plan)
    contexts = [int(x) for x in plan["contexts"]]

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Columns for the ML Model
    fieldnames = [
        "timestamp", 
        "engine", 
        "ctx_len",      # Feature 1 (Input size)
        "past_len",     # Feature 2 (KV Cache size)
        "run_id",       # Identifier
        "inference_idx",# Identifier (1st inference vs 100th)
        "latency_ms",   # TARGET VARIABLE
        "compute_ms"    # Secondary target (if needed)
    ]

    print(f"[INFO] generating ML dataset with {len(contexts)} buckets x {args.runs} runs x {args.iters} points...")
    print(f"[INFO] Output: {args.out}")

    with open(args.out, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Create a temporary file to hold the JSON dump for each run
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_json_path = tmp.name

        try:
            total_points = 0
            
            for ctx_len in contexts:
                past_len = max(0, ctx_len - 1)
                shapes = build_shapes_str(ctx_len)
                print(f"  [BUCKET] ctx_len={ctx_len}...", end="", flush=True)

                # Warmup run (optional, keeps GPU hot)
                try:
                    run_trtexec_and_export(args.engine, shapes, 10, args.warmup_ms, args.no_data_transfers, temp_json_path)
                except:
                    pass

                for run_id in range(args.runs):
                    # Run trtexec and dump to temp_json_path
                    run_trtexec_and_export(
                        engine_path=args.engine,
                        shapes=shapes,
                        iters=args.iters,
                        warmup_ms=args.warmup_ms,
                        no_data_transfers=args.no_data_transfers,
                        json_path=temp_json_path
                    )

                    # Parse the JSON dump
                    with open(temp_json_path, "r") as f:
                        raw_data = json.load(f)

                    # Extract every single inference point
                    # The JSON structure is usually: [ { "start":..., "end":..., "computeMs":... }, ... ]
                    for idx, entry in enumerate(raw_data):
                        # Calculate latency (End - Start)
                        # trtexec exports times in milliseconds (float)
                        # Sometimes structure is a list of dicts directly
                        
                        latency = entry.get('endMs', 0) - entry.get('startMs', 0)
                        compute = entry.get('computeMs', 0) # GPU only time

                        row = {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "engine": args.engine,
                            "ctx_len": ctx_len,
                            "past_len": past_len,
                            "run_id": run_id,
                            "inference_idx": idx,
                            "latency_ms": f"{latency:.5f}",
                            "compute_ms": f"{compute:.5f}"
                        }
                        writer.writerow(row)
                        total_points += 1
                
                print(f" Done. (Total: {total_points})")
                csvfile.flush()

        finally:
            # Cleanup temp file
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)

    print(f"\n✅ Dataset complete. Saved {total_points} rows to {args.out}")

if __name__ == "__main__":
    main()