#!/usr/bin/env python3
"""
Copy MLflow-logged models to src/serving/model/ directory structure.

This script finds the latest MLflow model runs and copies them to the serving
directory organized by threshold for easy model selection during inference.

Usage:
    python scripts/copy_models_to_serving.py [--threshold 0.25] [--run-id <run_id>]
"""

import os
import sys
import shutil
import argparse
import glob

# Make src importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

def find_latest_mlflow_model(experiment_name: str = None, threshold: float = None):
    """
    Find the latest MLflow model run.
    
    Args:
        experiment_name: Optional MLflow experiment name to filter by
        threshold: Optional threshold value to filter by
        
    Returns:
        Path to the model directory or None
    """
    mlruns_dir = os.path.join(project_root, "mlruns")
    if not os.path.exists(mlruns_dir):
        print(f"❌ MLflow runs directory not found: {mlruns_dir}")
        return None
    
    # Find all model artifacts
    model_paths = []
    for exp_dir in glob.glob(os.path.join(mlruns_dir, "*")):
        if not os.path.isdir(exp_dir) or exp_dir.endswith("meta.yaml"):
            continue
        
        # Check if this is an experiment directory
        meta_file = os.path.join(exp_dir, "meta.yaml")
        if os.path.exists(meta_file):
            # This is an experiment directory, look for runs
            for run_dir in glob.glob(os.path.join(exp_dir, "*")):
                if not os.path.isdir(run_dir):
                    continue
                
                model_path = os.path.join(run_dir, "artifacts", "model")
                if os.path.exists(model_path):
                    model_paths.append((run_dir, model_path))
    
    if not model_paths:
        print("❌ No MLflow models found in mlruns directory")
        return None
    
    # Sort by modification time (newest first)
    model_paths.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
    
    # Return the latest model
    latest_run_dir, latest_model_path = model_paths[0]
    print(f"✅ Found latest model: {latest_model_path}")
    return latest_model_path, latest_run_dir

def copy_model_to_serving(model_path: str, threshold: str, run_dir: str = None):
    """
    Copy MLflow model to serving directory structure.
    
    Args:
        model_path: Path to MLflow model directory
        threshold: Threshold value (e.g., "0.25")
        run_dir: Optional run directory to copy feature_columns.txt from
    """
    serving_model_dir = os.path.join(project_root, "src", "serving", "model", f"threshold_{threshold}")
    os.makedirs(serving_model_dir, exist_ok=True)
    
    # Copy the model directory
    target_model_path = os.path.join(serving_model_dir, "model")
    if os.path.exists(target_model_path):
        print(f"⚠️  Model already exists at {target_model_path}, removing...")
        shutil.rmtree(target_model_path)
    
    print(f"📦 Copying model from {model_path} to {target_model_path}...")
    shutil.copytree(model_path, target_model_path)
    print(f"✅ Model copied successfully to {target_model_path}")
    
    # Copy feature_columns.txt if available
    if run_dir:
        feature_file = os.path.join(run_dir, "artifacts", "feature_columns.txt")
        if os.path.exists(feature_file):
            target_feature_file = os.path.join(serving_model_dir, "feature_columns.txt")
            shutil.copy2(feature_file, target_feature_file)
            print(f"✅ Feature columns file copied to {target_feature_file}")

def main():
    parser = argparse.ArgumentParser(description="Copy MLflow models to serving directory")
    parser.add_argument("--threshold", type=str, default="0.25",
                       help="Threshold value for model (default: 0.25)")
    parser.add_argument("--run-id", type=str, default=None,
                       help="Specific MLflow run ID to copy (optional)")
    parser.add_argument("--all-thresholds", action="store_true",
                       help="Copy models for all common thresholds (0.22, 0.25, 0.3)")
    
    args = parser.parse_args()
    
    if args.all_thresholds:
        thresholds = ["0.22", "0.25", "0.3"]
        print(f"🔄 Copying models for all thresholds: {thresholds}")
        for threshold in thresholds:
            result = find_latest_mlflow_model(threshold=float(threshold))
            if result:
                model_path, run_dir = result
                copy_model_to_serving(model_path, threshold, run_dir)
                print()
    else:
        result = find_latest_mlflow_model(threshold=float(args.threshold))
        if result:
            model_path, run_dir = result
            copy_model_to_serving(model_path, args.threshold, run_dir)
        else:
            print("❌ Failed to find MLflow model")
            sys.exit(1)

if __name__ == "__main__":
    main()

