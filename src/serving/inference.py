"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import pandas as pd
import mlflow

# === MODEL LOADING CONFIGURATION ===
# IMPORTANT: This path is set during Docker container build
# In development: uses local MLflow artifacts
# In production: uses model copied to container at build time
MODEL_DIR = "/app/model"

# Default threshold for model selection (can be overridden via environment variable)
# Available thresholds: 0.22, 0.25, 0.3
DEFAULT_THRESHOLD = os.getenv("MODEL_THRESHOLD", "0.25")

# Dictionary to store loaded models by threshold
_models_cache = {}
_model_dirs_cache = {}

def _load_model_by_threshold(threshold: str = None):
    """
    Load model for a specific threshold from src/serving/model directory.
    
    Args:
        threshold: Model threshold (0.22, 0.25, or 0.3). If None, uses DEFAULT_THRESHOLD.
        
    Returns:
        Tuple of (model, model_dir) or (None, None) if loading fails
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    
    # Check cache first
    if threshold in _models_cache:
        return _models_cache[threshold], _model_dirs_cache[threshold]
    
    # Try loading from src/serving/model directory first (for local development)
    serving_dir = os.path.dirname(__file__)
    threshold_model_dir = os.path.join(serving_dir, "model", f"threshold_{threshold}")
    threshold_model_path = os.path.join(threshold_model_dir, "model")
    
    # Check if model directory exists (MLflow stores models in a 'model' subdirectory)
    if os.path.exists(threshold_model_path):
        try:
            # Load model from threshold-specific directory
            model = mlflow.pyfunc.load_model(threshold_model_path)
            _models_cache[threshold] = model
            _model_dirs_cache[threshold] = threshold_model_dir
            print(f"✅ Model loaded successfully for threshold {threshold} from {threshold_model_path}")
            return model, threshold_model_dir
        except Exception as e:
            print(f"⚠️  Warning: Failed to load model from {threshold_model_path}: {e}")
    elif os.path.exists(threshold_model_dir):
        # Try loading from the directory itself (if model is at root level)
        try:
            model = mlflow.pyfunc.load_model(threshold_model_dir)
            _models_cache[threshold] = model
            _model_dirs_cache[threshold] = threshold_model_dir
            print(f"✅ Model loaded successfully for threshold {threshold} from {threshold_model_dir}")
            return model, threshold_model_dir
        except Exception as e:
            print(f"⚠️  Warning: Failed to load model from {threshold_model_dir}: {e}")
    
    # Fallback to production path
    try:
        model = mlflow.pyfunc.load_model(MODEL_DIR)
        _models_cache[threshold] = model
        _model_dirs_cache[threshold] = MODEL_DIR
        print(f"✅ Model loaded successfully from {MODEL_DIR}")
        return model, MODEL_DIR
    except Exception as e:
        print(f"❌ Failed to load model from {MODEL_DIR}: {e}")
        # Fallback for local development (OPTIONAL)
        try:
            # Try loading from local MLflow tracking
            import glob
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            local_model_paths = (
                glob.glob(os.path.join(project_root, "mlruns/*/*/artifacts/model")) +
                glob.glob(os.path.join(project_root, "mlruns/*/models/*/artifacts"))
            )
            if local_model_paths:
                latest_model = max(local_model_paths, key=os.path.getmtime)
                model = mlflow.pyfunc.load_model(latest_model)
                _models_cache[threshold] = model
                _model_dirs_cache[threshold] = latest_model
                print(f"✅ Fallback: Loaded model from {latest_model}")
                return model, latest_model
            else:
                print("⚠️  Warning: No trained model found. API will start but predictions will fail.")
                print("   Train a model first using: python scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
                return None, None
        except Exception as fallback_error:
            print(f"⚠️  Warning: Fallback model loading failed: {fallback_error}")
            return None, None

# Load default model
model, MODEL_DIR = _load_model_by_threshold(DEFAULT_THRESHOLD)

# === FEATURE SCHEMA LOADING ===
# CRITICAL: Load the exact feature column order used during training
# This ensures the model receives features in the expected order
FEATURE_COLS = None
_feature_cols_cache = {}

def _load_feature_columns(model_dir: str = None):
    """Load feature columns from model directory or fallback locations."""
    if model_dir in _feature_cols_cache:
        return _feature_cols_cache[model_dir]
    
    if model_dir:
        try:
            feature_file = os.path.join(model_dir, "feature_columns.txt")
            if os.path.exists(feature_file):
                with open(feature_file) as f:
                    feature_cols = [ln.strip() for ln in f if ln.strip()]
                _feature_cols_cache[model_dir] = feature_cols
                print(f"✅ Loaded {len(feature_cols)} feature columns from {feature_file}")
                return feature_cols
        except Exception as e:
            print(f"⚠️  Warning: Could not load feature columns from {model_dir}: {e}")
    
    # Fallback: try to load from artifacts directory
    try:
        import json
        artifacts_file = os.path.join(os.path.dirname(__file__), "../../artifacts/feature_columns.json")
        if os.path.exists(artifacts_file):
            with open(artifacts_file) as f:
                feature_cols = json.load(f)
            print(f"✅ Loaded {len(feature_cols)} feature columns from artifacts")
            return feature_cols
    except Exception as e:
        print(f"⚠️  Warning: Could not load feature columns from artifacts: {e}")
    
    # Use default feature columns based on schema
    default_cols = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges", "MultipleLines_No phone service", "MultipleLines_Yes", "InternetService_Fiber optic", "InternetService_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes", "OnlineBackup_No internet service", "OnlineBackup_Yes", "DeviceProtection_No internet service", "DeviceProtection_Yes", "TechSupport_No internet service", "TechSupport_Yes", "StreamingTV_No internet service", "StreamingTV_Yes", "StreamingMovies_No internet service", "StreamingMovies_Yes", "Contract_One year", "Contract_Two year", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"]
    print(f"⚠️  Using default feature columns ({len(default_cols)} features)")
    return default_cols

# Load feature columns for default model
if MODEL_DIR:
    FEATURE_COLS = _load_feature_columns(MODEL_DIR)

# === FEATURE TRANSFORMATION CONSTANTS ===
# CRITICAL: These mappings must exactly match those used in training
# Any changes here will cause train/serve skew and degrade model performance

# Deterministic binary feature mappings (consistent with training)
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},           # Demographics
    "Partner": {"No": 0, "Yes": 1},               # Has partner
    "Dependents": {"No": 0, "Yes": 1},            # Has dependents  
    "PhoneService": {"No": 0, "Yes": 1},          # Phone service
    "PaperlessBilling": {"No": 0, "Yes": 1},      # Billing preference
}

# Numeric columns that need type coercion
NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame, feature_cols: list = None) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    
    This function is CRITICAL for production ML - it ensures that features are
    transformed exactly as they were during training to prevent train/serve skew.
    
    Transformation Pipeline:
    1. Clean column names and handle data types
    2. Apply deterministic binary encoding (using BINARY_MAP)
    3. One-hot encode remaining categorical features  
    4. Convert boolean columns to integers
    5. Align features with training schema and order
    
    Args:
        df: Single-row DataFrame with raw customer data
        
    Returns:
        DataFrame with features transformed and ordered for model input
        
    IMPORTANT: Any changes to this function must be reflected in training
    feature engineering to maintain consistency.
    """
    df = df.copy()
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    # === STEP 1: Numeric Type Coercion ===
    # Ensure numeric columns are properly typed (handle string inputs)
    for c in NUMERIC_COLS:
        if c in df.columns:
            # Convert to numeric, replacing invalid values with NaN
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # Fill NaN with 0 (same as training preprocessing)
            df[c] = df[c].fillna(0)
    
    # === STEP 2: Binary Feature Encoding ===
    # Apply deterministic mappings for binary features
    # CRITICAL: Must use exact same mappings as training
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)                    # Convert to string
                .str.strip()                    # Remove whitespace
                .map(mapping)                   # Apply binary mapping
                .astype("Int64")                # Handle NaN values
                .fillna(0)                      # Fill unknown values with 0
                .astype(int)                    # Final integer conversion
            )
    
    # === STEP 3: One-Hot Encoding for Remaining Categorical Features ===
    # Find remaining object/categorical columns (not in BINARY_MAP)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        # Apply one-hot encoding with drop_first=True (same as training)
        # This prevents multicollinearity by dropping the first category
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # === STEP 4: Boolean to Integer Conversion ===
    # Convert any boolean columns to integers (XGBoost compatibility)
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
    
    # === STEP 5: Feature Alignment with Training Schema ===
    # CRITICAL: Ensure features are in exact same order as training
    # Missing features get filled with 0, extra features are dropped
    # Use provided feature_cols parameter, or fall back to global FEATURE_COLS
    cols_to_use = feature_cols if feature_cols is not None else FEATURE_COLS
    if cols_to_use is not None:
        df = df.reindex(columns=cols_to_use, fill_value=0)
    
    return df

def predict(input_dict: dict, threshold: str = None) -> str:
    """
    Main prediction function for customer churn inference.
    
    This function provides the complete inference pipeline from raw customer data
    to business-friendly prediction output. It's called by both the FastAPI endpoint
    and the Gradio interface to ensure consistent predictions.
    
    Pipeline:
    1. Convert input dictionary to DataFrame
    2. Apply feature transformations (identical to training)
    3. Generate model prediction using loaded XGBoost model
    4. Convert prediction to user-friendly string
    
    Args:
        input_dict: Dictionary containing raw customer data with keys matching
                   the CustomerData schema (18 features total)
        threshold: Optional threshold to select specific model (0.22, 0.25, or 0.3).
                   If None, uses DEFAULT_THRESHOLD.
                   
    Returns:
        Human-readable prediction string:
        - "Likely to churn" for high-risk customers (model prediction = 1)
        - "Not likely to churn" for low-risk customers (model prediction = 0)
        
    Example:
        >>> customer_data = {
        ...     "gender": "Female", "tenure": 1, "Contract": "Month-to-month",
        ...     "MonthlyCharges": 85.0, ... # other features
        ... }
        >>> predict(customer_data)
        "Likely to churn"
        >>> predict(customer_data, threshold="0.22")
        "Likely to churn"
    """
    
    # Load model for specified threshold (or use default)
    pred_model, pred_model_dir = _load_model_by_threshold(threshold)
    
    if pred_model is None:
        raise Exception("Model not loaded. Please train a model first using: python scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Load feature columns for this model
    pred_feature_cols = _load_feature_columns(pred_model_dir)
    
    # === STEP 1: Convert Input to DataFrame ===
    # Create single-row DataFrame for pandas transformations
    df = pd.DataFrame([input_dict])
    
    # === STEP 2: Apply Feature Transformations ===
    # Use the same transformation pipeline as training
    # Pass feature columns directly to transform function
    df_enc = _serve_transform(df, feature_cols=pred_feature_cols)
    
    # === STEP 3: Generate Model Prediction ===
    # Call the loaded MLflow model for inference
    # The model returns predictions in various formats depending on the ML library
    try:
        preds = pred_model.predict(df_enc)
        
        # Normalize prediction output to consistent format
        if hasattr(preds, "tolist"):
            preds = preds.tolist()  # Convert numpy array to list
            
        # Extract single prediction value (for single-row input)
        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
            
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    # === STEP 4: Convert to Business-Friendly Output ===
    # Convert binary prediction (0/1) to actionable business language
    if result == 1:
        return "Likely to churn"      # High risk - needs intervention
    else:
        return "Not likely to churn"  # Low risk - maintain normal service