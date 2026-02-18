#!/usr/bin/env python3
"""
Step 7: Train Execution Time Predictors
Updated to predict 'compute_ms' (GPU Execution Time) since latency column was empty.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"  -> MAE:  {mae:.4f} ms")
    print(f"  -> RMSE: {rmse:.4f} ms")
    print(f"  -> R2:   {r2:.4f}")
    
    return preds, mae, r2

def main():
    # 1. Load Data
    csv_path = "training_data.csv"
    print(f"[INFO] Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: training_data.csv not found. Run measure_all.py first!")
        return

    # 2. Prepare Features (X) and Target (y)
    # We use 'ctx_len' to predict 'compute_ms' (Execution Time)
    X = df[["ctx_len"]] 
    y = df["compute_ms"]  # <--- CHANGED FROM latency_ms

    # Check for bad data
    if y.sum() == 0:
        print("[ERROR] 'compute_ms' column is also all zeros! Check your measurement script.")
        return

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Target Variable: GPU Compute Time (ms)")
    print(f"       Mean: {y.mean():.4f} ms | Max: {y.max():.4f} ms")
    
    # Split: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42),
        "KNN (k=5)": KNeighborsRegressor(n_neighbors=5)
    }

    results = {}
    
    # 4. Train & Evaluate Loop
    for name, model in models.items():
        preds, mae, r2 = train_and_evaluate(name, model, X_train, y_train, X_test, y_test)
        results[name] = {"preds": preds, "mae": mae, "r2": r2, "model": model}

    # 5. Best Model Selection
    best_name = min(results, key=lambda k: results[k]["mae"])
    print(f"\n🏆 Best Model: {best_name} (MAE: {results[best_name]['mae']:.4f} ms)")

    # 6. Visualization
    print("\n[INFO] Generating comparison plot...")
    plt.figure(figsize=(12, 6))
    
    # Plot real data (Test set only, to avoid clutter)
    plt.scatter(X_test, y_test, color="black", alpha=0.3, label="Actual Data", s=10)

    # Plot Model Predictions
    # Sort X_test for clean line plotting
    sorted_idx = np.argsort(X_test.values.flatten())
    X_sorted = X_test.values[sorted_idx]
    
    colors = {"Linear Regression": "red", "Random Forest": "blue", "XGBoost": "green", "KNN (k=5)": "orange"}
    
    for name, data in results.items():
        y_pred_sorted = data["preds"][sorted_idx]
        plt.plot(X_sorted, y_pred_sorted, label=f"{name} (R2={data['r2']:.2f})", color=colors[name], linewidth=2)

    plt.title(f"Model Comparison: Predicting Compute Time from Context Length\nBest: {best_name}")
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    out_file = "model_comparison.png"
    plt.savefig(out_file)
    print(f"✅ Plot saved to {out_file}")

if __name__ == "__main__":
    main()