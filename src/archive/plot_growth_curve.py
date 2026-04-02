#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def main():
    csv_path = "combined_dataset.csv"
    output_plot = "../plots/plot_growth_time_and_memory_labeled.png"
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(csv_path)
    for col in ['ctx_len', 'compute_ms', 'max_used_mem_bytes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Identify all unique context lengths to use as labels
    all_contexts = sorted(df['ctx_len'].unique())

    # 2. Setup Figure (Removed sharex=True to allow independent labels)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.3)

    # ---------------------------------------------------------
    # SUBPLOT 1: EXECUTION TIME
    # ---------------------------------------------------------
    df_time = df[df['compute_ms'] > 0].copy()
    sampled_time = df_time.groupby('ctx_len', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 10), random_state=42)
    )

    ax1.scatter(sampled_time['ctx_len'], sampled_time['compute_ms'], 
                alpha=0.4, color='royalblue', s=20, label='Inference Samples')

    mean_time = df_time.groupby('ctx_len')['compute_ms'].mean().reset_index()
    ax1.plot(mean_time['ctx_len'], mean_time['compute_ms'], 
             color='red', linewidth=3, label='Mean Latency', zorder=10)

    # Add X-Axis Labels to Top Plot
    ax1.set_xticks(all_contexts)
    ax1.set_xticklabels(all_contexts, rotation=45, fontsize=8)
    
    ax1.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Context Length (Tokens)", fontsize=10)
    ax1.set_title("GPT-2 Medium Performance: The Latency Cliff", fontsize=16, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left')

    # ---------------------------------------------------------
    # SUBPLOT 2: MEMORY USAGE
    # ---------------------------------------------------------
    df_mem = df[df['max_used_mem_bytes'] > 0].copy()
    df_mem['memory_gb'] = df_mem['max_used_mem_bytes'] / (1024**3)

    sampled_mem = df_mem.groupby('ctx_len', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 10), random_state=42)
    )

    ax2.scatter(sampled_mem['ctx_len'], sampled_mem['memory_gb'], 
                alpha=0.4, color='seagreen', s=20, label='Memory Samples')

    mean_mem = df_mem.groupby('ctx_len')['memory_gb'].mean().reset_index()
    ax2.plot(mean_mem['ctx_len'], mean_mem['memory_gb'], 
             color='darkgreen', linewidth=3, label='Mean VRAM/UVM Usage', zorder=10)

    # VRAM limit line (Matches your 2070 Super logs)
    vram_limit = 7.75 
    ax2.axhline(y=vram_limit, color='orange', linestyle='--', linewidth=2, label='Physical VRAM Limit')
    ax2.fill_between([0, max(all_contexts) + 1000], vram_limit, 16, color='orange', alpha=0.1, label='Oversubscription (UVM)')

    # Add X-Axis Labels to Bottom Plot
    ax2.set_xticks(all_contexts)
    ax2.set_xticklabels(all_contexts, rotation=45, fontsize=8)

    ax2.set_ylabel("Memory Usage (GB)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Context Length (Tokens)", fontsize=12, fontweight='bold')
    ax2.set_title("GPT-2 Medium Resource Usage: VRAM vs. Unified Memory", fontsize=14, pad=10)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, max(10, mean_mem['memory_gb'].max() + 1))

    # 3. Save
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✅ Labeled plot saved to {output_plot}")

if __name__ == "__main__":
    main()