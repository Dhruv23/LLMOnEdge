#!/usr/bin/env python3
"""
Step 6: Measurement harness (single-round decode) using trtexec.

What it does:
- Reads your single_round_plan.json (contexts list)
- For each context length bucket:
  - runs trtexec on your .plan engine using random inputs
  - repeats multiple runs per bucket (for more stable stats)
  - writes measurements.csv with mean latency + GPU compute time, etc.

Why trtexec:
- zero extra TensorRT/PyCUDA Python binding pain
- produces consistent timing breakdowns (latency, enqueue, H2D/D2H, GPU compute)
- supports dynamic shapes at inference time

Example:
python3 step6_measure.py \
  --engine engines/gpt2-medium-with-past.plan \
  --plan single_round_plan.json \
  --out measurements.csv \
  --iters 200 \
  --runs 5 \
  --warmup_ms 200 \
  --no_data_transfers
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

# GPT-2 Medium constants (matches your ONNX I/O shapes)
N_LAYERS = 24
N_HEADS = 16
HDIM = 64

RE_LATENCY = re.compile(r"Latency: .*mean\s*=\s*([0-9.]+)\s*ms")
RE_GPU = re.compile(r"GPU Compute Time: .*mean\s*=\s*([0-9.]+)\s*ms")
RE_ENQ = re.compile(r"Enqueue Time: .*mean\s*=\s*([0-9.]+)\s*ms")
RE_H2D = re.compile(r"H2D Latency: .*mean\s*=\s*([0-9.]+)\s*ms")
RE_D2H = re.compile(r"D2H Latency: .*mean\s*=\s*([0-9.]+)\s*ms")
RE_QPS = re.compile(r"Throughput:\s*([0-9.]+)\s*qps")


def load_plan(plan_path: str) -> dict:
    with open(plan_path, "r") as f:
        plan = json.load(f)
    if "contexts" not in plan or not plan["contexts"]:
        raise ValueError(f"{plan_path} must contain a non-empty 'contexts' list.")
    return plan


def build_shapes_str(ctx_len: int) -> str:
    """
    Builds the trtexec --shapes string for a single-round decode step at context length ctx_len.

    Single-round decode interpretation:
      - input_ids: 1x1 (the next token to decode)
      - position_ids: 1x1
      - attention_mask: 1 x ctx_len
      - past_key_values.*.{key,value}: 1 x 16 x (ctx_len-1) x 64
        (past length is ctx_len-1)
    """
    past_len = max(0, ctx_len - 1)

    parts = []
    parts.append(f"input_ids:1x1")
    parts.append(f"position_ids:1x1")
    parts.append(f"attention_mask:1x{ctx_len}")

    for i in range(N_LAYERS):
        parts.append(f"past_key_values.{i}.key:1x{N_HEADS}x{past_len}x{HDIM}")
        parts.append(f"past_key_values.{i}.value:1x{N_HEADS}x{past_len}x{HDIM}")

    return ",".join(parts)


def run_trtexec(engine_path: str, shapes: str, iters: int, warmup_ms: int, no_data_transfers: bool) -> str:
    """
    Runs trtexec once and returns stdout+stderr text.
    """
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes={shapes}",
        f"--iterations={iters}",
        "--duration=0",
        f"--warmUp={warmup_ms}",
        "--streams=1",
    ]

    # This flag is helpful if you want timing closer to "compute only".
    # If you want end-to-end (including H2D/D2H), omit it.
    if no_data_transfers:
        cmd.append("--noDataTransfers")

    # Keep logs parseable
    cmd.append("--verbose=0")

    # Run
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"trtexec failed (exit {proc.returncode}). Command:\n  {' '.join(cmd)}\n\nOutput:\n{proc.stdout}"
        )
    return proc.stdout


def parse_metrics(output_text: str) -> dict:
    """
    Pulls key mean metrics from trtexec output. Returns dict with floats or None if missing.
    """
    def grab(regex):
        m = regex.search(output_text)
        return float(m.group(1)) if m else None

    return {
        "latency_mean_ms": grab(RE_LATENCY),
        "gpu_compute_mean_ms": grab(RE_GPU),
        "enqueue_mean_ms": grab(RE_ENQ),
        "h2d_mean_ms": grab(RE_H2D),
        "d2h_mean_ms": grab(RE_D2H),
        "throughput_qps": grab(RE_QPS),
    }


def ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path to TensorRT engine (.plan)")
    ap.add_argument("--plan", required=True, help="Path to single_round_plan.json")
    ap.add_argument("--out", default="measurements.csv", help="Output CSV path")
    ap.add_argument("--iters", type=int, default=200, help="trtexec iterations per run (per bucket)")
    ap.add_argument("--runs", type=int, default=5, help="Number of independent runs per bucket")
    ap.add_argument("--warmup_ms", type=int, default=200, help="trtexec warmup time in ms")
    ap.add_argument("--no_data_transfers", action="store_true", help="Pass --noDataTransfers to trtexec")
    args = ap.parse_args()

    if not os.path.exists(args.engine):
        print(f"ERROR: engine not found: {args.engine}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.plan):
        print(f"ERROR: plan json not found: {args.plan}", file=sys.stderr)
        sys.exit(1)

    plan = load_plan(args.plan)
    contexts = [int(x) for x in plan["contexts"]]

    ensure_parent_dir(args.out)

    fieldnames = [
        "timestamp",
        "engine",
        "ctx_len",
        "past_len",
        "run_id",
        "iters",
        "warmup_ms",
        "no_data_transfers",
        "latency_mean_ms",
        "gpu_compute_mean_ms",
        "enqueue_mean_ms",
        "h2d_mean_ms",
        "d2h_mean_ms",
        "throughput_qps",
    ]

    write_header = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        print(f"[INFO] Engine: {args.engine}")
        print(f"[INFO] Context buckets: {contexts}")
        print(f"[INFO] Writing: {args.out}")
        print(f"[INFO] iters/run={args.iters}, runs/bucket={args.runs}, warmup_ms={args.warmup_ms}, noDataTransfers={args.no_data_transfers}")

        for ctx_len in contexts:
            past_len = max(0, ctx_len - 1)
            shapes = build_shapes_str(ctx_len)

            print(f"\n[BUCKET] ctx_len={ctx_len} (past_len={past_len})")

            # Optional: one quick “pre-warm” run per bucket to stabilize caches/tactics
            # (keeps your measured runs cleaner)
            try:
                _ = run_trtexec(args.engine, shapes, iters=min(50, args.iters), warmup_ms=args.warmup_ms, no_data_transfers=args.no_data_transfers)
            except Exception as e:
                print(f"[WARN] Pre-warm failed for ctx_len={ctx_len}: {e}", file=sys.stderr)

            for run_id in range(args.runs):
                tstamp = datetime.utcnow().isoformat() + "Z"
                print(f"  [RUN {run_id+1}/{args.runs}] running trtexec...")

                out_text = run_trtexec(
                    engine_path=args.engine,
                    shapes=shapes,
                    iters=args.iters,
                    warmup_ms=args.warmup_ms,
                    no_data_transfers=args.no_data_transfers,
                )
                metrics = parse_metrics(out_text)

                # Basic sanity: if latency missing, dump tail of output for debugging
                if metrics["latency_mean_ms"] is None:
                    tail = "\n".join(out_text.splitlines()[-40:])
                    raise RuntimeError(
                        f"Could not parse latency mean from trtexec output for ctx_len={ctx_len}.\n"
                        f"Output tail:\n{tail}"
                    )

                row = {
                    "timestamp": tstamp,
                    "engine": os.path.abspath(args.engine),
                    "ctx_len": ctx_len,
                    "past_len": past_len,
                    "run_id": run_id,
                    "iters": args.iters,
                    "warmup_ms": args.warmup_ms,
                    "no_data_transfers": int(args.no_data_transfers),
                    **metrics,
                }
                w.writerow(row)
                f.flush()

                print(
                    f"    latency_mean={metrics['latency_mean_ms']:.4f} ms, "
                    f"gpu_compute_mean={metrics['gpu_compute_mean_ms'] if metrics['gpu_compute_mean_ms'] is not None else 'NA'} ms, "
                    f"qps={metrics['throughput_qps'] if metrics['throughput_qps'] is not None else 'NA'}"
                )

    print("\n✅ Done. CSV written to:", args.out)
    print("Next: Step 7 (train predictor) can read measurements.csv directly.")


if __name__ == "__main__":
    main()
