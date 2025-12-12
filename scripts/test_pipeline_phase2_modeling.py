#!/usr/bin/env python3
"""
Test script for Phase 2: Model Training and Hyperparameter Optimization
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import optuna

# Setup paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Path to processed data
PROCESSED_DATA_PATH = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")

print("=== Phase 2: Modeling with XGBoost ===\n")

# Check if processed data exists
if not os.path.exists(PROCESSED_DATA_PATH):
    print(f"❌ Error: Processed data file not found at {PROCESSED_DATA_PATH}")
    print("   Please run test_pipeline_phase1_data_features.py first to create processed data.")
    sys.exit(1)

# Load processed data
print("[1] Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)
print(f"✅ Data loaded. Shape: {df.shape}")

# Ensure target is numeric 0/1
if "Churn" not in df.columns:
    print("❌ Error: 'Churn' column not found in data")
    sys.exit(1)

if df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"
print(f"✅ Churn values: {set(df['Churn'].unique())}")

# Prepare features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]
print(f"✅ Features: {X.shape[1]}, Samples: {X.shape[0]}")

# Train/test split
print("\n[2] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

THRESHOLD = 0.4
print(f"\n[3] Hyperparameter optimization (threshold={THRESHOLD})...")
print("   This may take a few minutes...")

def objective(trial):
    """Optuna objective function for hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= THRESHOLD).astype(int)
    return recall_score(y_test, y_pred, pos_label=1)

# Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\n[4] Results:")
print(f"✅ Best Params: {study.best_params}")
print(f"✅ Best Recall: {study.best_value:.4f}")

# Train final model with best params
print("\n[5] Training final model with best parameters...")
best_model = XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)

# Evaluate
proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (proba >= THRESHOLD).astype(int)

recall = recall_score(y_test, y_pred, pos_label=1)
precision = precision_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
roc_auc = roc_auc_score(y_test, proba)

print("\n[6] Final Model Performance:")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1 Score: {f1:.4f}")
print(f"   ROC AUC: {roc_auc:.4f}")

print("\n✅ Phase 2 pipeline completed successfully!")
