#!/usr/bin/env python3
"""
Step 6b-Streaming: Extended Stress Test (15k-30k)
Modified to run high-latency buckets one-by-one and stream results to CSV instantly.
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
import psutil
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
        self.cpu_samples = []
        self.ram_used_samples = []
        self.ram_util_samples = []
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

            # System CPU and RAM
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            sys_mem = psutil.virtual_memory()
            self.ram_used_samples.append(sys_mem.used)
            self.ram_util_samples.append(sys_mem.percent)

            time.sleep(self.poll_interval)

    def start(self):
        self.max_used_mem = 0
        self.peak_pcie_tx = 0
        self.peak_pcie_rx = 0
        self.cpu_samples = []
        self.ram_used_samples = []
        self.ram_util_samples = []
        # Prime the cpu_percent call
        psutil.cpu_percent(interval=None)
        self.keep_running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.keep_running = False
        if self.thread:
            self.thread.join()
        total_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total

        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0.0
        avg_ram_used = sum(self.ram_used_samples) / len(self.ram_used_samples) if self.ram_used_samples else 0
        avg_ram_util = sum(self.ram_util_samples) / len(self.ram_util_samples) if self.ram_util_samples else 0.0

        return {
            "max_used_mem_bytes": self.max_used_mem,
            "mem_utilization_pct": (self.max_used_mem / total_mem) * 100,
            "peak_pcie_tx_kbps": self.peak_pcie_tx,
            "peak_pcie_rx_kbps": self.peak_pcie_rx,
            "cpu_utilization_pct": avg_cpu,
            "sys_ram_used_bytes": int(avg_ram_used),
            "sys_ram_utilization_pct": avg_ram_util
        }

def build_shapes_str(ctx_len: int) -> str:
    past_len = max(0, ctx_len - 1)
    parts = [f"input_ids:1x1", f"position_ids:1x1", f"attention_mask:1x{ctx_len}"]
    for i in range(N_LAYERS):
        parts.append(f"past_key_values.{i}.key:1x{N_HEADS}x{past_len}x{HDIM}")
        parts.append(f"past_key_values.{i}.value:1x{N_HEADS}x{past_len}x{HDIM}")
    return ",".join(parts)

def run_single_inference(engine_path: str, shapes: str, json_path: str):
    """Runs a single iteration of trtexec to capture metrics immediately."""
    cmd = [
        "trtexec",
        f"--loadEngine={engine_path}",
        f"--shapes={shapes}",
        "--iterations=1",
        "--duration=0",
        "--warmUp=0",
        "--streams=1",
        "--verbose=0",
        f"--exportTimes={json_path}",
        "--useManagedMemory"
    ]

    monitor = GPUMonitor(device_idx=0)
    monitor.start()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    hardware_stats = monitor.stop()

    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed:\n{proc.stdout}")

    mem_metrics = {"device_exec_mem_bytes": 0, "device_ctx_mem_bytes": 0}
    exec_match = re.search(r'Total Device Execution Memory:\s*(\d+)', proc.stdout)
    if exec_match: mem_metrics["device_exec_mem_bytes"] = int(exec_match.group(1))
    ctx_match = re.search(r'Total Device Context Memory:\s*(\d+)', proc.stdout)
    if ctx_match: mem_metrics["device_ctx_mem_bytes"] = int(ctx_match.group(1))

    return mem_metrics, hardware_stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--out", default="training_dataset.csv")
    args = ap.parse_args()

    # New Test Points: 3 between 15k and 20k + a 30k point
    TARGET_CONTEXTS = [16250, 17500, 18750, 30000]
    ITERATIONS_PER_CTX = 10
    
    fieldnames = [
        "timestamp", "engine", "ctx_len", "past_len", "run_id", "inference_idx", 
        "latency_ms", "compute_ms", "device_exec_mem_bytes", "device_ctx_mem_bytes", 
        "max_used_mem_bytes", "mem_utilization_pct", "peak_pcie_tx_kbps", "peak_pcie_rx_kbps",
        "cpu_utilization_pct", "sys_ram_used_bytes", "sys_ram_utilization_pct"
    ]

    print(f"[INFO] Starting Extended Stress Test ({len(TARGET_CONTEXTS)} buckets x {ITERATIONS_PER_CTX} points)...")
    
    file_exists = os.path.exists(args.out)
    with open(args.out, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            temp_json_path = tmp.name

        try:
            for ctx_len in TARGET_CONTEXTS:
                print(f"\n[BUCKET] ctx_len={ctx_len}")
                shapes = build_shapes_str(ctx_len)
                
                for i in range(ITERATIONS_PER_CTX):
                    print(f"  [RUN {i+1}/{ITERATIONS_PER_CTX}]...", end="", flush=True)
                    start_time = time.time()
                    
                    try:
                        mem_metrics, hw_stats = run_single_inference(args.engine, shapes, temp_json_path)
                        
                        with open(temp_json_path, "r") as f:
                            raw_data = json.load(f)
                        
                        entry = raw_data[0]
                        latency = entry.get('endMs', 0) - entry.get('startMs', 0)
                        compute = entry.get('computeMs', 0)

                        row = {
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "engine": args.engine,
                            "ctx_len": ctx_len,
                            "past_len": ctx_len - 1,
                            "run_id": 1000, # unique ID for this extended run
                            "inference_idx": i,
                            "latency_ms": f"{latency:.5f}",
                            "compute_ms": f"{compute:.5f}",
                            "device_exec_mem_bytes": mem_metrics["device_exec_mem_bytes"],
                            "device_ctx_mem_bytes": mem_metrics["device_ctx_mem_bytes"],
                            "max_used_mem_bytes": hw_stats["max_used_mem_bytes"],
                            "mem_utilization_pct": f"{hw_stats['mem_utilization_pct']:.2f}",
                            "peak_pcie_tx_kbps": hw_stats["peak_pcie_tx_kbps"],
                            "peak_pcie_rx_kbps": hw_stats["peak_pcie_rx_kbps"],
                            "cpu_utilization_pct": f"{hw_stats['cpu_utilization_pct']:.2f}",
                            "sys_ram_used_bytes": hw_stats["sys_ram_used_bytes"],
                            "sys_ram_utilization_pct": f"{hw_stats['sys_ram_utilization_pct']:.2f}"
                        }
                        writer.writerow(row)
                        csvfile.flush()
                        
                        elapsed = time.time() - start_time
                        print(f" Done ({elapsed:.2f}s, Latency: {latency:.2f}ms)")
                        
                    except Exception as e:
                        print(f" FAILED: {e}")

        finally:
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)

    print(f"\n✅ Extended stress test complete.")

if __name__ == "__main__":
    main()