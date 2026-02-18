#!/usr/bin/env python3
"""
Step 8: Train Median Predictor
Aggregates the raw noisy data into Medians to find the 'true' trend.
Generates 'model_comparison_median.png'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    # Train
    model.fit(X_train, y_train)
    # Predict
    preds = model.predict(X_test)
    # Score
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return preds, mae, r2

def main():
    # 1. Load Raw Data
    csv_path = "training_data.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: training_data.csv not found.")
        return

    # 2. Augment: Group by Context Length and calculate Median
    print(f"[INFO] Raw data: {len(df)} rows. Aggregating by Median...")
    
    # This creates a small, clean dataset (e.g., 20 rows for 20 buckets)
    df_median = df.groupby("ctx_len")["compute_ms"].median().reset_index()
    
    X = df_median[["ctx_len"]]
    y = df_median["compute_ms"]

    print(f"[INFO] Median dataset shape: {df_median.shape}")
    print(f"       (This represents the 'Typical' performance curve)")

    # 3. Train Models on the Median Data
    # Since data is small (20 points), we use a smaller test size or just fit all to see the trend.
    # But to be rigorous, we'll still split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42),
        "KNN (k=3)": KNeighborsRegressor(n_neighbors=3) # Lower k for smaller dataset
    }

    results = {}
    print("\n--- Training Models on Median Data ---")
    for name, model in models.items():
        preds, mae, r2 = train_and_evaluate(name, model, X_train, y_train, X_test, y_test)
        results[name] = {"preds": preds, "mae": mae, "r2": r2}
        print(f"{name:>18} | R2: {r2:.4f} | MAE: {mae:.4f} ms")

    # 4. Generate the Clean Plot
    print("\n[INFO] Generating median comparison plot...")
    plt.figure(figsize=(10, 6))
    
    # A. Plot the aggregated Median points (The "Ground Truth" for this experiment)
    plt.scatter(X, y, color="black", label="Actual Medians", s=50, zorder=5)

    # B. Plot the Model Predictions (lines)
    # We use the whole X range to draw smooth lines
    X_range = np.linspace(X["ctx_len"].min(), X["ctx_len"].max(), 100).reshape(-1, 1)
    
    colors = {"Linear Regression": "red", "Random Forest": "blue", "XGBoost": "green", "KNN (k=3)": "orange"}
    
    for name, model in models.items():
        # Train on FULL median dataset for the plot lines (to look nicest)
        model.fit(X, y) 
        y_pred_line = model.predict(X_range)
        plt.plot(X_range, y_pred_line, label=f"{name}", color=colors[name], linewidth=2, alpha=0.8)

    plt.title("Predicting Median Execution Time (Noise Filtered)")
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("Median Execution Time (ms)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    out_file = "model_comparison_median.png"
    plt.savefig(out_file)
    print(f"✅ Plot saved to {out_file}")

if __name__ == "__main__":
    main()