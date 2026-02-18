#!/usr/bin/env python3
"""
Step 6: Extended Measurement Harness & Plotter
- Runs trtexec to gather statistics (Min, Max, Median, P99).
- Automatically cleans old plots and generates 3 meaningful charts.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import shutil
from datetime import datetime

# --- IMPORTS FOR PLOTTING ---
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARN] pandas or matplotlib not found. Plotting will be skipped.")

# GPT-2 Medium constants
N_LAYERS = 24
N_HEADS = 16
HDIM = 64

# Regex patterns for detailed stats
RE_LATENCY_MEAN = re.compile(r"Latency:.*mean\s*=\s*([0-9.]+)\s*ms")
RE_LATENCY_MIN  = re.compile(r"Latency:.*min\s*=\s*([0-9.]+)\s*ms")
RE_LATENCY_MAX  = re.compile(r"Latency:.*max\s*=\s*([0-9.]+)\s*ms")
RE_LATENCY_MED  = re.compile(r"Latency:.*median\s*=\s*([0-9.]+)\s*ms")
RE_LATENCY_P99  = re.compile(r"Latency:.*percentile\(99%\)\s*=\s*([0-9.]+)\s*ms")
RE_GPU = re.compile(r"GPU Compute Time: .*mean\s*=\s*([0-9.]+)\s*ms")
RE_QPS = re.compile(r"Throughput:\s*([0-9.]+)\s*qps")


def load_plan(plan_path: str) -> dict:
    with open(plan_path, "r") as f:
        plan = json.load(f)
    if "contexts" not in plan or not plan["contexts"]:
        raise ValueError(f"{plan_path} must contain a non-empty 'contexts' list.")
    return plan


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


def run_trtexec(engine_path: str, shapes: str, iters: int, warmup_ms: int, 
                no_data_transfers: bool, export_json_path: str = None) -> str:
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes={shapes}",
        f"--iterations={iters}",
        "--duration=0",
        f"--warmUp={warmup_ms}",
        "--streams=1",
        "--verbose=0"
    ]

    if no_data_transfers:
        cmd.append("--noDataTransfers")

    if export_json_path:
        cmd.append(f"--exportTimes={export_json_path}")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed: {proc.stdout}")
        
    return proc.stdout


def parse_metrics(output_text: str) -> dict:
    def grab(regex):
        m = regex.search(output_text)
        return float(m.group(1)) if m else None

    return {
        "latency_mean_ms": grab(RE_LATENCY_MEAN),
        "latency_min_ms":  grab(RE_LATENCY_MIN),
        "latency_max_ms":  grab(RE_LATENCY_MAX),
        "latency_median_ms": grab(RE_LATENCY_MED),
        "latency_p99_ms":  grab(RE_LATENCY_P99),
        "gpu_compute_mean_ms": grab(RE_GPU),
        "throughput_qps": grab(RE_QPS),
    }


def generate_plots(csv_path: str):
    """Generates 3 clean, meaningful plots from the CSV data."""
    if not PLOTTING_AVAILABLE:
        return

    print("\n[PLOTS] Cleaning old plots and generating new ones...")
    
    # 1. Clean Plot Directory
    plot_dir = "../plots"
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

    # 2. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV for plotting: {e}")
        return

    # Ensure we sort by context length so lines connect correctly
    df = df.sort_values(by="ctx_len")
    x = df["ctx_len"]

    # --- PLOT 1: Latency Scaling (The most important one) ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, df["latency_median_ms"], label="Median Latency", color="blue", linewidth=2)
    plt.plot(x, df["latency_p99_ms"], label="P99 Latency (Worst Case)", color="red", linestyle="--")
    plt.title("Latency vs Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.savefig(f"{plot_dir}/1_latency_scaling.png")
    plt.close()
    print(f"   -> Saved {plot_dir}/1_latency_scaling.png")

    # --- PLOT 2: Throughput (Tokens per Second) ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, df["throughput_qps"], color="green", linewidth=2, marker="o", markersize=4)
    plt.title("Throughput vs Context Length")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Throughput (Queries/Sec)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"{plot_dir}/2_throughput.png")
    plt.close()
    print(f"   -> Saved {plot_dir}/2_throughput.png")

    # --- PLOT 3: Stability (Min/Max Band) ---
    # This shows if the engine is "jittery" or stable
    plt.figure(figsize=(10, 6))
    plt.plot(x, df["latency_median_ms"], color="black", label="Median")
    plt.fill_between(x, df["latency_min_ms"], df["latency_max_ms"], color="blue", alpha=0.2, label="Min-Max Range")
    plt.title("Latency Stability (Jitter Analysis)")
    plt.xlabel("Context Length (tokens)")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(f"{plot_dir}/3_stability_analysis.png")
    plt.close()
    print(f"   -> Saved {plot_dir}/3_stability_analysis.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default="measurements_detailed.csv")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--warmup_ms", type=int, default=200)
    ap.add_argument("--no_data_transfers", action="store_true")
    ap.add_argument("--save_raw", action="store_true")
    
    args = ap.parse_args()

    plan = load_plan(args.plan)
    contexts = [int(x) for x in plan["contexts"]]

    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        
    # Clean raw dump directory if needed
    raw_dir = "raw_traces"
    if args.save_raw:
        if os.path.exists(raw_dir):
            shutil.rmtree(raw_dir)
        os.makedirs(raw_dir)

    fieldnames = [
        "timestamp", "engine", "ctx_len", "run_id", "iters", 
        "latency_min_ms", "latency_max_ms", "latency_mean_ms", 
        "latency_median_ms", "latency_p99_ms", 
        "gpu_compute_mean_ms", "throughput_qps"
    ]

    # CHANGED TO "w" TO OVERWRITE OLD FILES AUTOMATICALLY
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        print(f"[INFO] Running {args.iters} datapoints per bucket.")
        
        for ctx_len in contexts:
            shapes = build_shapes_str(ctx_len)
            print(f"\n[BUCKET] ctx_len={ctx_len}")

            # Pre-warm
            try:
                run_trtexec(args.engine, shapes, iters=50, warmup_ms=args.warmup_ms, 
                           no_data_transfers=args.no_data_transfers)
            except:
                pass

            for run_id in range(args.runs):
                json_path = None
                if args.save_raw:
                    json_path = os.path.join(raw_dir, f"trace_ctx{ctx_len}_run{run_id}.json")

                print(f"  [RUN {run_id+1}/{args.runs}] Collecting {args.iters} datapoints...")
                
                out_text = run_trtexec(
                    engine_path=args.engine,
                    shapes=shapes,
                    iters=args.iters,
                    warmup_ms=args.warmup_ms,
                    no_data_transfers=args.no_data_transfers,
                    export_json_path=json_path
                )
                
                metrics = parse_metrics(out_text)

                if metrics["latency_mean_ms"] is None:
                    print(f"[ERROR] Failed to parse metrics for ctx={ctx_len}")
                    continue

                row = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "engine": args.engine,
                    "ctx_len": ctx_len,
                    "run_id": run_id,
                    "iters": args.iters,
                    **metrics
                }
                w.writerow(row)
                f.flush()

                print(f"    Median: {metrics['latency_median_ms']} ms | P99: {metrics['latency_p99_ms']} ms")

    print(f"\n✅ Data collection done. CSV written to: {args.out}")
    
    # --- TRIGGER PLOT GENERATION AUTOMATICALLY ---
    generate_plots(args.out)

if __name__ == "__main__":
    main()