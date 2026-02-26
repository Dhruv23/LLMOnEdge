#!/usr/bin/env python3
"""
Step 8: Train Percentile Predictors (Large Scale)
Processes 200,000+ data points, trains ML models on various statistical slices,
and outputs individual + combined plots with actual data markers and metric labels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return preds, mae, r2

def get_best_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
    }
    
    best_name, best_model = None, None
    best_mae = float('inf')
    best_r2 = -float('inf')
    
    for name, model in models.items():
        _, mae, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        if mae < best_mae:
            best_mae, best_r2 = mae, r2
            best_name, best_model = name, model
            
    # Retrain best model on the FULL dataset for smooth plotting
    best_model.fit(X, y)
    return best_model, best_name, best_mae, best_r2

def main():
    # 1. Setup Output Directory
    plot_dir = "../plots/ML_PLOTS"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"[INFO] Plot directory ready at: {plot_dir}")

    # 2. Load Raw Data (All 200,000 rows)
    csv_path = "training_dataset_large.csv" 
    try:
        print(f"[INFO] Loading {csv_path}...")
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] {csv_path} not found.")
        return

    if df["compute_ms"].sum() == 0:
        print("[ERROR] 'compute_ms' column is all zeros!")
        return

    print(f"[INFO] Successfully loaded {len(df)} data points.")

    # 3. Define the Percentiles
    percentiles = {
        "Min": 0.0,
        "5th": 0.05,
        "10th": 0.10,
        "25th": 0.25,
        "Median": 0.50,
        "65th": 0.65,
        "75th": 0.75,
        "85th": 0.85,
        "95th": 0.95,
        "Max": 1.0
    }

    print(f"[INFO] Aggregating data into {len(percentiles)} statistical slices...")
    
    # Calculate quantiles
    agg_funcs = {name: pd.NamedAgg(column="compute_ms", aggfunc=lambda x, q=q: x.quantile(q)) for name, q in percentiles.items()}
    df_agg = df.groupby("ctx_len").agg(**agg_funcs).reset_index()

    # 4. Train Models
    # Dictionary to store the model AND its metrics for labeling
    trained_models = {}
    print("\n--- Training Results ---")
    print(f"{'Target Slice':<12} | {'Best Model':<20} | {'MAE (ms)':<8} | {'R2':<6}")
    print("-" * 55)

    X_agg = df_agg[["ctx_len"]]
    
    # Raw Model
    X_raw, y_raw = df[["ctx_len"]], df["compute_ms"]
    best_raw_model, best_raw_name, raw_mae, raw_r2 = get_best_model(X_raw, y_raw)
    trained_models["Raw Data"] = {"model": best_raw_model, "mae": raw_mae, "r2": raw_r2}
    print(f"{'Raw Data':<12} | {best_raw_name:<20} | {raw_mae:<8.4f} | {raw_r2:<6.4f}")

    # Percentile Models
    for slice_name in percentiles.keys():
        y_slice = df_agg[slice_name]
        best_model, best_name, best_mae, best_r2 = get_best_model(X_agg, y_slice)
        trained_models[slice_name] = {"model": best_model, "mae": best_mae, "r2": best_r2}
        print(f"{slice_name:<12} | {best_name:<20} | {best_mae:<8.4f} | {best_r2:<6.4f}")

    # 5. Generate Individual Plots
    print("\n[INFO] Generating individual plots...")
    X_range = np.linspace(df["ctx_len"].min(), df["ctx_len"].max(), 100).reshape(-1, 1)
    X_range_df = pd.DataFrame(X_range, columns=["ctx_len"])

    for slice_name in list(percentiles.keys()) + ["Raw Data"]:
        plt.figure(figsize=(10, 6))
        
        # Plot all 200,000 raw points in the background
        plt.scatter(df["ctx_len"], df["compute_ms"], color="gray", alpha=0.02, s=1, label="All 200k Raw Points", zorder=1)

        model_info = trained_models[slice_name]
        model = model_info["model"]
        mae = model_info["mae"]
        r2 = model_info["r2"]
        
        y_pred_line = model.predict(X_range_df)
        
        color = "red" if slice_name in ["Min", "Max", "95th"] else "blue"
        
        # Construct label with metrics
        line_label = f"{slice_name} Predictor (R²: {r2:.3f}, MAE: {mae:.3f}ms)"
        plt.plot(X_range, y_pred_line, label=line_label, color=color, linewidth=3, zorder=5)

        # Plot the exact aggregated points it trained on
        if slice_name != "Raw Data":
            plt.scatter(df_agg["ctx_len"], df_agg[slice_name], color="black", s=30, label=f"Actual {slice_name} Values", zorder=6)

        plt.title(f"GPT-2 GPU Execution Time: {slice_name} Predictor", fontsize=14, pad=15)
        plt.xlabel("Context Length (Tokens)", fontsize=12)
        plt.ylabel("Execution Time (ms)", fontsize=12)
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.6)
        
        safe_name = slice_name.replace(" ", "_")
        out_file = os.path.join(plot_dir, f"plot_{safe_name}.png")
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close()

    # 6. Generate Master Combined Plot
    print("[INFO] Generating master combined plot...")
    plt.figure(figsize=(16, 9)) # Slightly wider to accommodate larger legend
    
    # Background raw data
    plt.scatter(df["ctx_len"], df["compute_ms"], color="gray", alpha=0.02, s=1, label="Raw Data (200k pts)", zorder=1)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in np.linspace(0, 1, len(percentiles))]

    for i, slice_name in enumerate(percentiles.keys()):
        model_info = trained_models[slice_name]
        model = model_info["model"]
        mae = model_info["mae"]
        r2 = model_info["r2"]
        
        y_pred_line = model.predict(X_range_df)
        
        # Plot predictor line with metrics in label
        line_label = f"{slice_name} (R²: {r2:.3f}, MAE: {mae:.3f}ms)"
        plt.plot(X_range, y_pred_line, label=line_label, color=colors[i], linewidth=2.5, zorder=5)
        
        # Plot the actual aggregated raw data points for this line (as 'x' markers)
        plt.scatter(df_agg["ctx_len"], df_agg[slice_name], color=colors[i], marker='x', s=40, zorder=6, alpha=0.8)

    plt.title("GPT-2 Execution Time Predictors: All Percentiles Combined", fontsize=16, pad=15)
    plt.xlabel("Context Length (Tokens)", fontsize=12)
    plt.ylabel("Execution Time (ms)", fontsize=12)
    
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Predictors & Metrics")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Use bbox_inches="tight" to ensure the external legend isn't cut off
    out_file = os.path.join(plot_dir, "plot_ALL_COMBINED.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✅ All {len(percentiles) + 2} plots saved to {plot_dir}")

if __name__ == "__main__":
    main()