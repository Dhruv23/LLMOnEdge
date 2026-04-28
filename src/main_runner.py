import threading
import time
import cupy as cp
import torch.cuda.nvtx as nvtx
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt

# --- 1. Corunner Thread Setup ---
class CorunnerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stop_event = threading.Event()
        self.throughput_data = [] # Store (start_time, loop_duration)
        
    def run(self):
        stream = cp.cuda.Stream(non_blocking=True)
        
        # NVTX marker for the background thread initialization
        nvtx.range_push("Corunner_Init")
        with stream:
            # Allocate matrices
            a = cp.random.rand(1024, 1024, dtype=cp.float32)
            b = cp.random.rand(1024, 1024, dtype=cp.float32)
            nvtx.range_pop()
            
            while not self.stop_event.is_set():
                start_time = time.perf_counter()
                
                # Interference workload
                c = cp.matmul(a, b)
                d = cp.sin(c)
                stream.synchronize() 
                
                end_time = time.perf_counter()
                self.throughput_data.append((start_time, end_time - start_time))

    def stop(self):
        self.stop_event.set()

# --- 2. Main Execution and Profiling ---

def profile_time_gating():
    print("Loading vLLM Engine...")
    llm = LLM(
        model="gpt2-medium", 
        enforce_eager=True,
        gpu_memory_utilization=0.6,
        max_num_seqs=1,
        max_model_len=1024
    )

    print("Starting Corunner Thread...")
    corunner = CorunnerThread()
    corunner.start()
    
    try:
        # Stabilization pause
        time.sleep(2) 
        
        # === PHASE A: ISOLATED PREFILL ===
        print("\n--- Executing Isolated Prefill (~512 tokens in, 1 token out) ---")
        prefill_prompt = "The quick brown fox jumps over the lazy dog. " * 50
        prefill_params = SamplingParams(max_tokens=1, temperature=0.0)
        
        nvtx.range_push("Phase_A_Isolated_Prefill")
        t0_prefill = time.perf_counter()
        llm.generate([prefill_prompt], prefill_params)
        t1_prefill = time.perf_counter()
        nvtx.range_pop()
        
        # Pause between runs to establish a clean baseline in the plot
        time.sleep(2)

        # === PHASE B: ISOLATED DECODE ===
        print("\n--- Executing Isolated Decode (1 token in, 128 tokens out) ---")
        decode_prompt = "Hello" 
        decode_params = SamplingParams(max_tokens=128, temperature=0.0)
        
        nvtx.range_push("Phase_B_Isolated_Decode")
        t0_decode = time.perf_counter()
        llm.generate([decode_prompt], decode_params)
        t1_decode = time.perf_counter()
        nvtx.range_pop()

        time.sleep(1)

    finally:
        print("\nStopping Corunner...")
        corunner.stop()
        corunner.join()

    # --- 3. Extracting and Normalizing Data ---
    print("Processing Data...")
    
    timestamps = [data[0] for data in corunner.throughput_data]
    latencies = [data[1] for data in corunner.throughput_data]
    
    base_time = timestamps[0]
    timestamps = [t - base_time for t in timestamps]
    
    t0_prefill_norm = t0_prefill - base_time
    t1_prefill_norm = t1_prefill - base_time
    t0_decode_norm = t0_decode - base_time
    t1_decode_norm = t1_decode - base_time

    # --- 4. Plot 1: The Full Timeline ---
    print("Generating Full Timeline Plot...")
    plt.figure(figsize=(14, 6))
    
    plt.plot(timestamps, latencies, color='black', alpha=0.6, linewidth=1, label='Corunner Loop Latency')
    plt.scatter(timestamps, latencies, color='black', s=2)

    plt.axvspan(t0_prefill_norm, t1_prefill_norm, color='red', alpha=0.3, label='LLM Prefill (Compute-Bound)')
    plt.axvspan(t0_decode_norm, t1_decode_norm, color='blue', alpha=0.3, label='LLM Decode (Memory-Bound)')

    plt.title('Corunner Interference Timeline: Prefill vs Decode Gating', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Corunner Loop Duration (seconds)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("layer_level_interference.png", dpi=300)
    
    # --- 5. Plot 2: The Decode Zoom (Jitter Analysis) ---
    print("Generating Decode Zoom Plot...")
    
    # Filter data points that only fall within the Decode window
    decode_mask = [(t >= t0_decode_norm) and (t <= t1_decode_norm) for t in timestamps]
    decode_times = [t for t, m in zip(timestamps, decode_mask) if m]
    decode_lats = [l for l, m in zip(latencies, decode_mask) if m]
    
    plt.figure(figsize=(12, 5))
    
    # Using a line plot with distinct markers to highlight the oscillation
    plt.plot(decode_times, decode_lats, color='#4B0082', alpha=0.8, linewidth=1.5, marker='o', markersize=3, label='Corunner Iteration')
    
    plt.title('Decode Phase Analysis: Iteration-Level Scheduling (Sawtooth Pattern)', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Corunner Loop Duration (seconds)', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("decode_jitter_zoom.png", dpi=300)
    print("Plots successfully saved: 'layer_level_interference.png' and 'decode_jitter_zoom.png'")

if __name__ == "__main__":
    profile_time_gating()