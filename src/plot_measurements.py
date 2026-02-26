#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = "training_dataset_large.csv"
    out_dir = "../plots"
    
    print(f"[INFO] Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {csv_path}. Make sure you are in the correct directory.")
        return

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory set to: {out_dir}")

    print("[INFO] Aggregating metrics (Min, Median, Max, P99)...")
    # Group by context length to get all required statistical bounds
    agg_df = df.groupby("ctx_len")["compute_ms"].agg(
        Min='min',
        Median='median',
        Max='max',
        P99=lambda x: x.quantile(0.99)
    ).reset_index()

    # Calculate Throughput (Queries per Second)
    # 1000 ms / Median Latency (ms) = Queries per second
    agg_df['Throughput'] = 1000.0 / agg_df['Median']

    # ---------------------------------------------------------
    # Plot 1: Latency vs Context Length (Median vs P99)
    # ---------------------------------------------------------
    print("[INFO] Generating 'Latency vs Context Length' plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(agg_df["ctx_len"], agg_df["Median"], color='blue', linestyle='-', linewidth=2, label='Median Latency')
    plt.plot(agg_df["ctx_len"], agg_df["P99"], color='red', linestyle='--', linewidth=2, label='P99 Latency (Worst Case)')
    
    plt.title("Latency vs Context Length", fontsize=14)
    plt.xlabel("Context Length (tokens)", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(out_dir, "Latency_vs_ContextLength_P99.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # Plot 2: Latency Stability (Jitter Analysis)
    # ---------------------------------------------------------
    print("[INFO] Generating 'Latency Stability (Jitter Analysis)' plot...")
    plt.figure(figsize=(10, 6))
    
    # Fill the area between Min and Max to show the full jitter range
    plt.fill_between(agg_df["ctx_len"], agg_df["Min"], agg_df["Max"], color='blue', alpha=0.2, label='Min-Max Range')
    
    # Plot the Median as a solid black line on top
    plt.plot(agg_df["ctx_len"], agg_df["Median"], color='black', linestyle='-', linewidth=1.5, label='Median')
    
    plt.title("Latency Stability (Jitter Analysis)", fontsize=14)
    plt.xlabel("Context Length (tokens)", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    
    plt.savefig(os.path.join(out_dir, "Latency_Stability_Jitter.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------
    # Plot 3: Throughput vs Context Length
    # ---------------------------------------------------------
    print("[INFO] Generating 'Throughput vs Context Length' plot...")
    plt.figure(figsize=(10, 6))
    
    # Plot throughput with a green line and dot markers
    plt.plot(agg_df["ctx_len"], agg_df["Throughput"], color='green', marker='o', markersize=4, linestyle='-', linewidth=2)
    
    plt.title("Throughput vs Context Length", fontsize=14)
    plt.xlabel("Context Length (tokens)", fontsize=12)
    plt.ylabel("Throughput (Queries/Sec)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    
    plt.savefig(os.path.join(out_dir, "Throughput_vs_ContextLength.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ All 3 plots successfully generated and saved to {out_dir}/")

if __name__ == "__main__":
    main()