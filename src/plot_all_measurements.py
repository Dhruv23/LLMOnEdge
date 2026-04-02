#!/usr/bin/env python3
"""
plot_all_measurements.py

Consolidated script to generate:
1. Standard latency aggregation plots (Median vs P99, Jitter, Throughput)
2. Predictor training & plots for execution time & memory (Linear Regression & XGBoost)
3. Growth curve ("Latency Cliff" and VRAM limit oversubscription)
4. New Memory Oversubscription plots:
   a. Saturation Point (mem_utilization_pct vs ctx_len)
   b. Oversubscription Signature (peak_pcie_tx_kbps & peak_pcie_rx_kbps vs ctx_len)
   c. Latency Cost (compute_ms vs ctx_len colored by mem_utilization_pct and peak_pcie_tx_kbps)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

def train_isolated_models(X, y):
    """Trains both LR and XGB on the data, evaluates, and retrains on full dataset for plotting."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)
    lr_model.fit(X, y) # Retrain for plotting

    # 2. XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)
    xgb_model.fit(X, y) # Retrain for plotting

    return {
        "LR": {"model": lr_model, "mae": lr_mae, "r2": lr_r2},
        "XGB": {"model": xgb_model, "mae": xgb_mae, "r2": xgb_r2}
    }

def run_pipeline_for_target(df_raw, target_col, target_label, unit, percentiles, plot_dir):
    """Runs the aggregation, training, and plotting pipeline for a specific target column."""

    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n{'='*50}")
    print(f"🚀 RUNNING PIPELINE FOR: {target_label.upper()}")
    print(f"{'='*50}")

    # Aggregate Data
    print(f"[INFO] Aggregating data into {len(percentiles)} statistical slices...")
    agg_funcs = {name: pd.NamedAgg(column=target_col, aggfunc=lambda x, q=q: x.quantile(q)) for name, q in percentiles.items()}
    df_agg = df_raw.groupby("ctx_len").agg(**agg_funcs).reset_index()

    # Train Models
    trained_models = {}
    print("\n--- Training Results ---")
    print(f"{'Slice':<10} | {'LR MAE':<8} | {'LR R2':<8} | {'XGB MAE':<8} | {'XGB R2':<8}")
    print("-" * 60)

    X_agg = df_agg[["ctx_len"]]
    X_range = np.linspace(df_raw["ctx_len"].min(), df_raw["ctx_len"].max(), 100).reshape(-1, 1)
    X_range_df = pd.DataFrame(X_range, columns=["ctx_len"])

    # Raw Model Training
    trained_models["Raw Data"] = train_isolated_models(df_raw[["ctx_len"]], df_raw[target_col])
    res = trained_models["Raw Data"]
    print(f"{'Raw Data':<10} | {res['LR']['mae']:<8.4f} | {res['LR']['r2']:<8.4f} | {res['XGB']['mae']:<8.4f} | {res['XGB']['r2']:<8.4f}")

    # Percentile Models Training
    for slice_name in percentiles.keys():
        trained_models[slice_name] = train_isolated_models(X_agg, df_agg[slice_name])
        res = trained_models[slice_name]
        print(f"{slice_name:<10} | {res['LR']['mae']:<8.4f} | {res['LR']['r2']:<8.4f} | {res['XGB']['mae']:<8.4f} | {res['XGB']['r2']:<8.4f}")

    # Generate Individual Plots
    print(f"\n[INFO] Generating individual {target_label} plots...")
    for slice_name in list(percentiles.keys()) + ["Raw Data"]:
        plt.figure(figsize=(10, 6))

        # Background Raw Data
        plt.scatter(df_raw["ctx_len"], df_raw[target_col], color="gray", alpha=0.05, s=1, label="Raw Data Points", zorder=1)

        res = trained_models[slice_name]

        # Plot LR
        lr_label = f"Linear Regression (R²: {res['LR']['r2']:.3f}, MAE: {res['LR']['mae']:.3f})"
        plt.plot(X_range, res['LR']['model'].predict(X_range_df), label=lr_label, color="blue", linewidth=2.5, linestyle="--", zorder=5)

        # Plot XGB
        xgb_label = f"XGBoost (R²: {res['XGB']['r2']:.3f}, MAE: {res['XGB']['mae']:.3f})"
        plt.plot(X_range, res['XGB']['model'].predict(X_range_df), label=xgb_label, color="red", linewidth=2.5, zorder=5)

        # Plot actual aggregated points
        if slice_name != "Raw Data":
            plt.scatter(df_agg["ctx_len"], df_agg[slice_name], color="black", s=40, label=f"Actual {slice_name} Targets", zorder=6)

        plt.title(f"GPT-2 {target_label}: {slice_name} Predictor", fontsize=14, pad=15)
        plt.xlabel("Context Length (Tokens)", fontsize=12)
        plt.ylabel(f"{target_label} ({unit})", fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)

        safe_name = slice_name.replace(" ", "_")
        plt.savefig(os.path.join(plot_dir, f"{safe_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()

    # Generate Master Combined Plots (One for LR, One for XGB)
    for model_type, color_map in [("LR", "winter"), ("XGB", "autumn")]:
        plt.figure(figsize=(16, 9))
        plt.scatter(df_raw["ctx_len"], df_raw[target_col], color="gray", alpha=0.02, s=1, label="Raw Data", zorder=1)

        cmap = plt.get_cmap(color_map)
        colors = [cmap(i) for i in np.linspace(0, 1, len(percentiles))]

        for i, slice_name in enumerate(percentiles.keys()):
            res = trained_models[slice_name][model_type]
            line_label = f"{slice_name} (R²: {res['r2']:.3f})"
            plt.plot(X_range, res['model'].predict(X_range_df), label=line_label, color=colors[i], linewidth=2, zorder=5)
            plt.scatter(df_agg["ctx_len"], df_agg[slice_name], color=colors[i], marker='x', s=30, zorder=6, alpha=0.8)

        plt.title(f"GPT-2 {target_label}: All Percentiles ({model_type} Models)", fontsize=16, pad=15)
        plt.xlabel("Context Length (Tokens)", fontsize=12)
        plt.ylabel(f"{target_label} ({unit})", fontsize=12)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title=f"{model_type} Predictors")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.savefig(os.path.join(plot_dir, f"ALL_COMBINED_{model_type}.png"), dpi=200, bbox_inches="tight")
        plt.close()

def plot_standard_latency(df, out_dir):
    print("[INFO] Generating standard latency plots...")
    os.makedirs(out_dir, exist_ok=True)
    df_valid = df[df["compute_ms"] > 0].copy()

    agg_df = df_valid.groupby("ctx_len")["compute_ms"].agg(
        Min='min',
        Median='median',
        Max='max',
        P99=lambda x: x.quantile(0.99)
    ).reset_index()

    agg_df['Throughput'] = 1000.0 / agg_df['Median']

    # Plot 1: Latency vs Context Length
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

    # Plot 2: Latency Stability
    plt.figure(figsize=(10, 6))
    plt.fill_between(agg_df["ctx_len"], agg_df["Min"], agg_df["Max"], color='blue', alpha=0.2, label='Min-Max Range')
    plt.plot(agg_df["ctx_len"], agg_df["Median"], color='black', linestyle='-', linewidth=1.5, label='Median')
    plt.title("Latency Stability (Jitter Analysis)", fontsize=14)
    plt.xlabel("Context Length (tokens)", fontsize=12)
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, "Latency_Stability_Jitter.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 3: Throughput vs Context Length
    plt.figure(figsize=(10, 6))
    plt.plot(agg_df["ctx_len"], agg_df["Throughput"], color='green', marker='o', markersize=4, linestyle='-', linewidth=2)
    plt.title("Throughput vs Context Length", fontsize=14)
    plt.xlabel("Context Length (tokens)", fontsize=12)
    plt.ylabel("Throughput (Queries/Sec)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(out_dir, "Throughput_vs_ContextLength.png"), dpi=200, bbox_inches="tight")
    plt.close()

def plot_growth_curve(df, out_dir):
    print("[INFO] Generating Growth Curve plot...")
    output_plot = os.path.join(out_dir, "plot_growth_time_and_memory_labeled.png")

    all_contexts = sorted(df['ctx_len'].unique())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.3)

    # SUBPLOT 1: EXECUTION TIME
    df_time = df[df['compute_ms'] > 0].copy()

    # Select up to 10 samples per context length
    # Using groupby.head(10) is a robust way to get a subset without messing up indices
    sampled_time = df_time.sample(frac=1, random_state=42).groupby('ctx_len').head(10)

    ax1.scatter(sampled_time['ctx_len'], sampled_time['compute_ms'],
                alpha=0.4, color='royalblue', s=20, label='Inference Samples')

    mean_time = df_time.groupby('ctx_len')['compute_ms'].mean().reset_index()
    ax1.plot(mean_time['ctx_len'], mean_time['compute_ms'],
             color='red', linewidth=3, label='Mean Latency', zorder=10)

    ax1.set_xticks(all_contexts)
    ax1.set_xticklabels(all_contexts, rotation=45, fontsize=8)
    ax1.set_ylabel("Execution Time (ms)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Context Length (Tokens)", fontsize=10)
    ax1.set_title("GPT-2 Medium Performance: The Latency Cliff", fontsize=16, pad=10)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left')

    # SUBPLOT 2: MEMORY USAGE
    if "max_used_mem_bytes" in df.columns:
        df_mem = df[df['max_used_mem_bytes'] > 0].copy()
        if not df_mem.empty:
            df_mem['memory_gb'] = df_mem['max_used_mem_bytes'] / (1024**3)

            sampled_mem = df_mem.sample(frac=1, random_state=42).groupby('ctx_len').head(10)

            ax2.scatter(sampled_mem['ctx_len'], sampled_mem['memory_gb'],
                        alpha=0.4, color='seagreen', s=20, label='Memory Samples')

            mean_mem = df_mem.groupby('ctx_len')['memory_gb'].mean().reset_index()
            ax2.plot(mean_mem['ctx_len'], mean_mem['memory_gb'],
                     color='darkgreen', linewidth=3, label='Mean VRAM/UVM Usage', zorder=10)

            vram_limit = 7.75
            ax2.axhline(y=vram_limit, color='orange', linestyle='--', linewidth=2, label='Physical VRAM Limit')
            ax2.fill_between([0, max(all_contexts) + 1000], vram_limit, max(16, mean_mem['memory_gb'].max() + 1), color='orange', alpha=0.1, label='Oversubscription (UVM)')

            ax2.set_xticks(all_contexts)
            ax2.set_xticklabels(all_contexts, rotation=45, fontsize=8)

            ax2.set_ylabel("Memory Usage (GB)", fontsize=12, fontweight='bold')
            ax2.set_xlabel("Context Length (Tokens)", fontsize=12, fontweight='bold')
            ax2.set_title("GPT-2 Medium Resource Usage: VRAM vs. Unified Memory", fontsize=14, pad=10)
            ax2.grid(True, linestyle='--', alpha=0.3)
            ax2.legend(loc='upper left')
            ax2.set_ylim(0, max(10, mean_mem['memory_gb'].max() + 1))

    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=200, bbox_inches='tight')
    plt.close()

def plot_system_resource_metrics(df, out_dir):
    print("[INFO] Generating System Resource plots (CPU/RAM)...")
    os.makedirs(out_dir, exist_ok=True)
    all_contexts = sorted(df['ctx_len'].unique())

    # Filter out -1 data
    df_valid = df[(df['cpu_utilization_pct'] > -1) &
                  (df['sys_ram_used_bytes'] > -1) &
                  (df['sys_ram_utilization_pct'] > -1)].copy()

    if df_valid.empty:
        print("[WARNING] No valid data for system resource metrics (> -1). Skipping...")
        return

    # 1. CPU Utilization vs Context Length
    plt.figure(figsize=(12, 6))
    mean_cpu = df_valid.groupby('ctx_len')['cpu_utilization_pct'].mean().reset_index()

    # Scatter samples
    sampled_cpu = df_valid.sample(frac=1, random_state=42).groupby('ctx_len').head(10)
    plt.scatter(sampled_cpu['ctx_len'], sampled_cpu['cpu_utilization_pct'], alpha=0.4, color='orange', s=20)
    plt.plot(mean_cpu['ctx_len'], mean_cpu['cpu_utilization_pct'], color='darkred', linewidth=2, label='Mean CPU Utilization')

    plt.title("CPU Utilization vs Context Length", fontsize=14)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("CPU Utilization (%)", fontsize=12)
    plt.xticks(all_contexts, rotation=45, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(out_dir, "CPU_Utilization_vs_ContextLength.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. System RAM Usage vs Context Length
    plt.figure(figsize=(12, 6))
    df_valid['sys_ram_gb'] = df_valid['sys_ram_used_bytes'] / (1024**3)
    mean_ram = df_valid.groupby('ctx_len')['sys_ram_gb'].mean().reset_index()

    # Scatter samples
    sampled_ram = df_valid.sample(frac=1, random_state=42).groupby('ctx_len').head(10)
    plt.scatter(sampled_ram['ctx_len'], sampled_ram['sys_ram_gb'], alpha=0.4, color='cyan', s=20)
    plt.plot(mean_ram['ctx_len'], mean_ram['sys_ram_gb'], color='teal', linewidth=2, label='Mean System RAM Usage')

    plt.title("System RAM Usage vs Context Length", fontsize=14)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("System RAM (GB)", fontsize=12)
    plt.xticks(all_contexts, rotation=45, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(out_dir, "System_RAM_Usage_vs_ContextLength.png"), dpi=200, bbox_inches='tight')
    plt.close()

def plot_oversubscription_metrics(df, out_dir):
    print("[INFO] Generating Oversubscription plots...")
    os.makedirs(out_dir, exist_ok=True)
    all_contexts = sorted(df['ctx_len'].unique())

    # Filter out -1 data
    df_valid = df[(df['mem_utilization_pct'] > -1) &
                  (df['peak_pcie_tx_kbps'] > -1) &
                  (df['peak_pcie_rx_kbps'] > -1) &
                  (df['compute_ms'] > 0)].copy()

    if df_valid.empty:
        print("[WARNING] No valid data for oversubscription metrics (> -1). Skipping...")
        return

    # 1. Saturation Point (mem_utilization_pct vs ctx_len)
    plt.figure(figsize=(12, 6))
    mean_util = df_valid.groupby('ctx_len')['mem_utilization_pct'].mean().reset_index()

    # Scatter samples
    sampled_util = df_valid.sample(frac=1, random_state=42).groupby('ctx_len').head(10)
    plt.scatter(sampled_util['ctx_len'], sampled_util['mem_utilization_pct'], alpha=0.4, color='purple', s=20)
    plt.plot(mean_util['ctx_len'], mean_util['mem_utilization_pct'], color='indigo', linewidth=2, label='Mean Mem Utilization')

    plt.axhline(y=100, color='red', linestyle='--', linewidth=2, label='GPU VRAM Ceiling (100%)')

    plt.title("Memory Saturation Point", fontsize=14)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("Memory Utilization (%)", fontsize=12)
    plt.xticks(all_contexts, rotation=45, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(out_dir, "Saturation_Point.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Oversubscription Signature (PCIe Bandwidth vs ctx_len)
    plt.figure(figsize=(12, 6))

    mean_tx = df_valid.groupby('ctx_len')['peak_pcie_tx_kbps'].mean().reset_index()
    mean_rx = df_valid.groupby('ctx_len')['peak_pcie_rx_kbps'].mean().reset_index()

    plt.plot(mean_tx['ctx_len'], mean_tx['peak_pcie_tx_kbps'], color='darkorange', linewidth=2, label='Peak PCIe TX (kbps)')
    plt.plot(mean_rx['ctx_len'], mean_rx['peak_pcie_rx_kbps'], color='dodgerblue', linewidth=2, label='Peak PCIe RX (kbps)')

    plt.title("Oversubscription Signature: PCIe Bandwidth Spikes", fontsize=14)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("Bandwidth (kbps)", fontsize=12)
    plt.xticks(all_contexts, rotation=45, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(out_dir, "Oversubscription_Signature.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 3. Latency Cost Colored by Utilization and TX Bandwidth
    # Plot A: Colored by Mem Utilization
    plt.figure(figsize=(12, 6))
    scatter1 = plt.scatter(df_valid['ctx_len'], df_valid['compute_ms'],
                           c=df_valid['mem_utilization_pct'], cmap='viridis',
                           alpha=0.7, s=20, edgecolors='none')
    cbar1 = plt.colorbar(scatter1)
    cbar1.set_label("Memory Utilization (%)")

    plt.title("Latency Cost Driven by Memory Saturation", fontsize=14)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.xticks(all_contexts, rotation=45, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(out_dir, "Latency_Cost_MemUtil.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # Plot B: Colored by Peak PCIe TX
    plt.figure(figsize=(12, 6))
    scatter2 = plt.scatter(df_valid['ctx_len'], df_valid['compute_ms'],
                           c=df_valid['peak_pcie_tx_kbps'], cmap='viridis',
                           alpha=0.7, s=20, edgecolors='none')
    cbar2 = plt.colorbar(scatter2)
    cbar2.set_label("Peak PCIe TX (kbps)")

    plt.title("Latency Cost Driven by PCIe Thrashing", fontsize=14)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    plt.xticks(all_contexts, rotation=45, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.savefig(os.path.join(out_dir, "Latency_Cost_PCIeTX.png"), dpi=200, bbox_inches='tight')
    plt.close()


def main():
    base_plot_dir = "../plots/jules"
    os.makedirs(base_plot_dir, exist_ok=True)

    # 1. Load Data
    csv_path = "combined_dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = "training_dataset_large.csv"

    try:
        print(f"[INFO] Loading {csv_path}...")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {csv_path} or training_dataset_large.csv")
        return

    for col in ['ctx_len', 'compute_ms', 'max_used_mem_bytes', 'mem_utilization_pct', 'peak_pcie_tx_kbps', 'peak_pcie_rx_kbps', 'cpu_utilization_pct', 'sys_ram_used_bytes', 'sys_ram_utilization_pct']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    percentiles = {
        "Min": 0.0, "5th": 0.05, "10th": 0.10, "25th": 0.25,
        "Median": 0.50, "65th": 0.65, "75th": 0.75, "85th": 0.85,
        "95th": 0.95, "Max": 1.0
    }

    # Pipeline 1: Execution Time Predictors
    if "compute_ms" in df.columns:
        compute_dir = os.path.join(base_plot_dir, "ML_PLOTS", "Compute_Time")
        run_pipeline_for_target(
            df_raw=df[df['compute_ms'] > 0],
            target_col="compute_ms",
            target_label="Execution Time",
            unit="ms",
            percentiles=percentiles,
            plot_dir=compute_dir
        )

    # Pipeline 2: Memory Predictors
    if "max_used_mem_bytes" in df.columns:
        df_mem = df[df["max_used_mem_bytes"] > 0].copy()
        if not df_mem.empty:
            df_mem["memory_mb"] = df_mem["max_used_mem_bytes"] / (1024 * 1024)
            mem_dir = os.path.join(base_plot_dir, "ML_PLOTS", "Memory_Usage")
            run_pipeline_for_target(
                df_raw=df_mem,
                target_col="memory_mb",
                target_label="Dynamic Memory Usage",
                unit="MB",
                percentiles=percentiles,
                plot_dir=mem_dir
            )

    # Generate standard plots
    plot_standard_latency(df, base_plot_dir)

    # Generate Growth Curve
    plot_growth_curve(df, base_plot_dir)

    # Generate Oversubscription plots
    plot_oversubscription_metrics(df, base_plot_dir)

    # Generate System Resource plots (CPU/RAM)
    plot_system_resource_metrics(df, base_plot_dir)

    print(f"\n✅ All script executions finished. Check the {base_plot_dir} folder.")

if __name__ == "__main__":
    main()
