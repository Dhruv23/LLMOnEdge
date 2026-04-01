#!/usr/bin/env python3
"""
Step 8: Train Percentile Predictors (Compute & Memory)
Trains Linear Regression and XGBoost in isolation on various statistical slices 
for BOTH Execution Time and Memory Behavior. Generates comparative plots.
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

def main():
    # 1. Setup Base Directories
    base_plot_dir = "../plots/ML_PLOTS"
    os.makedirs(base_plot_dir, exist_ok=True)
    
    # 2. Load Raw Data
    # Assuming user is working with the combined output or the new large one
    csv_path = "combined_dataset.csv" 
    if not os.path.exists(csv_path):
        csv_path = "training_dataset_large.csv" # Fallback
        
    try:
        print(f"[INFO] Loading {csv_path}...")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Could not find {csv_path}. Please check filename.")
        return

    # 3. Define Percentiles
    percentiles = {
        "Min": 0.0, "5th": 0.05, "10th": 0.10, "25th": 0.25, 
        "Median": 0.50, "65th": 0.65, "75th": 0.75, "85th": 0.85, 
        "95th": 0.95, "Max": 1.0
    }

    # ---------------------------------------------------------
    # PIPELINE 1: EXECUTION TIME
    # ---------------------------------------------------------
    if "compute_ms" in df.columns:
        compute_dir = os.path.join(base_plot_dir, "Compute_Time")
        run_pipeline_for_target(
            df_raw=df,
            target_col="compute_ms",
            target_label="Execution Time",
            unit="ms",
            percentiles=percentiles,
            plot_dir=compute_dir
        )

    # ---------------------------------------------------------
    # PIPELINE 2: MEMORY BEHAVIOR
    # ---------------------------------------------------------
    # We use max_used_mem_bytes for dynamic footprint. 
    # Must filter out the -1 dummy data first so we don't skew the models!
    if "max_used_mem_bytes" in df.columns:
        df_mem = df[df["max_used_mem_bytes"] > 0].copy()
        
        if len(df_mem) > 0:
            # Convert bytes to Megabytes for readable plots
            df_mem["memory_mb"] = df_mem["max_used_mem_bytes"] / (1024 * 1024)
            
            mem_dir = os.path.join(base_plot_dir, "Memory_Usage")
            run_pipeline_for_target(
                df_raw=df_mem,
                target_col="memory_mb",
                target_label="Dynamic Memory Usage",
                unit="MB",
                percentiles=percentiles,
                plot_dir=mem_dir
            )
        else:
            print("[WARNING] Memory column found, but no valid data (>0) exists. Skipping memory plots.")

    print(f"\n✅ All script executions finished. Check the {base_plot_dir} folder.")

if __name__ == "__main__":
    main()