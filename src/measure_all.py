#!/usr/bin/env python3
"""
Step 6b: ML Dataset Generator (Memory Profiling + Schema Migration)
Runs trtexec and exports EVERY single inference latency to a CSV.
If an older CSV exists, it backfills old rows with -1 for the new memory metrics.

Output Format (dataset.csv):
timestamp, engine, ctx_len, past_len, run_id, inference_idx, latency_ms, compute_ms, 
device_exec_mem_bytes, device_ctx_mem_bytes, max_used_mem_bytes, mem_utilization_pct, 
peak_pcie_tx_kbps, peak_pcie_rx_kbps
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import re
import time
import threading
import pynvml
from datetime import datetime

# GPT-2 Medium constants
N_LAYERS = 24
N_HEADS = 16
HDIM = 64

class GPUMonitor:
    def __init__(self, device_idx=0, poll_interval=0.05):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        self.poll_interval = poll_interval
        self.keep_running = False
        self.max_used_mem = 0
        self.peak_pcie_tx = 0
        self.peak_pcie_rx = 0
        self.thread = None

    def _monitor_loop(self):
        while self.keep_running:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            if mem_info.used > self.max_used_mem:
                self.max_used_mem = mem_info.used

            tx = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            rx = pynvml.nvmlDeviceGetPcieThroughput(self.handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            
            if tx > self.peak_pcie_tx: self.peak_pcie_tx = tx
            if rx > self.peak_pcie_rx: self.peak_pcie_rx = rx

            time.sleep(self.poll_interval)

    def start(self):
        self.max_used_mem = 0
        self.peak_pcie_tx = 0
        self.peak_pcie_rx = 0
        self.keep_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.keep_running = False
        if self.thread:
            self.thread.join()
        
        total_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
        
        return {
            "max_used_mem_bytes": self.max_used_mem,
            "mem_utilization_pct": (self.max_used_mem / total_mem) * 100,
            "peak_pcie_tx_kbps": self.peak_pcie_tx,
            "peak_pcie_rx_kbps": self.peak_pcie_rx
        }

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
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes={shapes}",
        f"--iterations={iters}",
        "--duration=0",
        f"--warmUp={warmup_ms}",
        "--streams=1",
        "--verbose=0",
        f"--exportTimes={json_path}", 
        "--useManagedMemory"  # Use unified memory to reduce PCIe transfers
    ]

    if no_data_transfers:
        cmd.append("--noDataTransfers")

    monitor = GPUMonitor(device_idx=0, poll_interval=0.05)
    monitor.start()

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    hardware_stats = monitor.stop()

    if proc.returncode != 0:
        if "out of memory" in proc.stdout.lower():
            print(f"\n[WARNING] OOM Crash detected! Hardware stats right before crash: {hardware_stats}")
        raise RuntimeError(f"trtexec failed:\n{proc.stdout}")

    mem_metrics = {"device_exec_mem_bytes": 0, "device_ctx_mem_bytes": 0}
    
    exec_match = re.search(r'Total Device Execution Memory:\s*(\d+)', proc.stdout)
    if exec_match:
        mem_metrics["device_exec_mem_bytes"] = int(exec_match.group(1))
        
    ctx_match = re.search(r'Total Device Context Memory:\s*(\d+)', proc.stdout)
    if ctx_match:
        mem_metrics["device_ctx_mem_bytes"] = int(ctx_match.group(1))

    return mem_metrics, hardware_stats

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

    # Define Full Schema
    fieldnames = [
        "timestamp", "engine", "ctx_len", "past_len", "run_id", "inference_idx", 
        "latency_ms", "compute_ms", "device_exec_mem_bytes", "device_ctx_mem_bytes", 
        "max_used_mem_bytes", "mem_utilization_pct", "peak_pcie_tx_kbps", "peak_pcie_rx_kbps"
    ]
    new_memory_fields = [
        "device_exec_mem_bytes", "device_ctx_mem_bytes", "max_used_mem_bytes", 
        "mem_utilization_pct", "peak_pcie_tx_kbps", "peak_pcie_rx_kbps"
    ]

    # --- MIGRATION BLOCK ---
    if os.path.exists(args.out):
        with open(args.out, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_headers = next(reader, [])
        
        # Check if we are missing any of the new columns
        if set(fieldnames) - set(existing_headers):
            print(f"[INFO] Legacy CSV detected. Migrating {args.out} to new schema...")
            temp_out = args.out + ".tmp"
            
            with open(args.out, 'r', newline='') as f_in, open(temp_out, 'w', newline='') as f_out:
                reader = csv.DictReader(f_in)
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
                
                rows_migrated = 0
                for row in reader:
                    # Fill all missing memory fields with -1 (Dummy value for ML)
                    for field in new_memory_fields:
                        if field not in row:
                            row[field] = -1
                    writer.writerow(row)
                    rows_migrated += 1
            
            # Safely replace the old file with the new one
            os.replace(temp_out, args.out)
            print(f"[INFO] Migration complete. Updated {rows_migrated} rows with -1 dummy data.")
        else:
            print("[INFO] Existing dataset already uses the correct schema.")

    # Determine file mode: append if it exists, otherwise write new
    file_mode = "a" if os.path.exists(args.out) else "w"

    print(f"[INFO] Generating ML dataset with {len(contexts)} buckets x {args.runs} runs x {args.iters} points...")
    print(f"[INFO] Output: {args.out} (Mode: {file_mode})")

    # --- MAIN GENERATION LOOP ---
    with open(args.out, file_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_json_path = tmp.name

        try:
            total_points = 0
            
            for ctx_len in contexts:
                past_len = max(0, ctx_len - 1)
                shapes = build_shapes_str(ctx_len)
                print(f"  [BUCKET] ctx_len={ctx_len}...", end="", flush=True)

                try:
                    run_trtexec_and_export(args.engine, shapes, 10, args.warmup_ms, args.no_data_transfers, temp_json_path)
                except:
                    pass

                for run_id in range(args.runs):
                    mem_metrics, hw_stats = run_trtexec_and_export(
                        engine_path=args.engine,
                        shapes=shapes,
                        iters=args.iters,
                        warmup_ms=args.warmup_ms,
                        no_data_transfers=args.no_data_transfers,
                        json_path=temp_json_path
                    )

                    with open(temp_json_path, "r") as f:
                        raw_data = json.load(f)

                    for idx, entry in enumerate(raw_data):
                        latency = entry.get('endMs', 0) - entry.get('startMs', 0)
                        compute = entry.get('computeMs', 0)

                        row = {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "engine": args.engine,
                            "ctx_len": ctx_len,
                            "past_len": past_len,
                            "run_id": run_id,
                            "inference_idx": idx,
                            "latency_ms": f"{latency:.5f}",
                            "compute_ms": f"{compute:.5f}",
                            "device_exec_mem_bytes": mem_metrics["device_exec_mem_bytes"],
                            "device_ctx_mem_bytes": mem_metrics["device_ctx_mem_bytes"],
                            "max_used_mem_bytes": hw_stats["max_used_mem_bytes"],
                            "mem_utilization_pct": f"{hw_stats['mem_utilization_pct']:.2f}",
                            "peak_pcie_tx_kbps": hw_stats["peak_pcie_tx_kbps"],
                            "peak_pcie_rx_kbps": hw_stats["peak_pcie_rx_kbps"]
                        }
                        writer.writerow(row)
                        total_points += 1
                
                print(f" Appended. (+{total_points} new points so far)")
                csvfile.flush()

        finally:
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)

    print(f"\n✅ Dataset update complete. Appended {total_points} new rows to {args.out}")

if __name__ == "__main__":
    main()