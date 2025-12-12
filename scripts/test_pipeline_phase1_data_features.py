#!/usr/bin/env python3
"""
Test script for Phase 1: Data Loading → Preprocessing → Feature Engineering
"""
import os
import sys

# Make sure Python can find your src package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# === CONFIG ===
DATA_PATH = os.path.join(project_root, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET_COL = "Churn"


def main():
    """Run Phase 1 pipeline test."""
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===\n")

    # 1. Load Data
    print("[1] Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data file not found at {DATA_PATH}")
        return
    
    df = load_data(DATA_PATH)
    print(f"✅ Data loaded. Shape: {df.shape}")
    print(f"   Columns: {list(df.columns[:5])}...")
    print(df.head(3))

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"✅ Data after preprocessing. Shape: {df_clean.shape}")
    print(f"   Columns: {list(df_clean.columns[:5])}...")
    print(df_clean.head(3))

    # 3. Build Features
    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"✅ Data after feature engineering. Shape: {df_features.shape}")
    print(f"   Columns: {list(df_features.columns[:5])}...")
    print(df_features.head(3))

    # Validation checks
    print("\n[4] Validation checks...")
    assert "Churn" in df_features.columns, "Churn column should be preserved"
    assert df_features.shape[0] == df.shape[0], "Number of rows should be preserved"
    assert df_features.shape[1] >= df_clean.shape[1], "Should have more features after engineering"
    
    # Check Churn is 0/1
    if "Churn" in df_features.columns:
        churn_values = set(df_features["Churn"].dropna().unique())
        assert churn_values <= {0, 1}, f"Churn should be 0/1, got {churn_values}"
        print(f"✅ Churn values: {churn_values}")

    print("\n✅ Phase 1 pipeline completed successfully!")


if __name__ == "__main__":
    main()
