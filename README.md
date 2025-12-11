# Churn Prediction Project

An end-to-end machine learning pipeline for predicting customer churn in telecommunications companies. This project demonstrates a complete ML workflow from exploratory data analysis to model deployment, including feature engineering, hyperparameter tuning, model interpretability, and risk scoring.

## 📋 Use Case

**Problem Statement:** Predict which customers are likely to churn (cancel their service) so that the business can proactively engage with at-risk customers and implement retention strategies.

**Business Value:**
- Reduce customer churn by identifying at-risk customers early
- Optimize retention campaign targeting and resource allocation
- Improve customer lifetime value (CLV) through proactive intervention
- Enable data-driven decision making for customer retention strategies

**Dataset:** Telco Customer Churn Dataset
- **Size:** 7,043 customers with 21 features
- **Target:** Binary classification (Churn: Yes/No)
- **Churn Rate:** 26.54% (imbalanced dataset)

## 📚 Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                          # Original dataset
│   ├── features/                     # Feature-engineered data
│   └── processed/                    # Train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature Engineering
│   ├── 03_model_training.ipynb      # Baseline Model Training
│   ├── 04_hyperparameter_tuning.ipynb # Hyperparameter Optimization
│   └── 05_model_interpretability.ipynb # Model Interpretation & Risk Scoring
├── models/                           # Trained models and artifacts
├── src/
│   ├── config.py                     # Project configuration
│   ├── models/
│   │   └── risk_scoring.py          # Risk scoring module
│   └── api/                          # API endpoints (to be implemented)
├── mlruns/                           # MLflow experiment tracking
└── pyproject.toml                    # Project dependencies
```

## 🔬 Notebooks Overview

### 1. **01_eda.ipynb** - Exploratory Data Analysis
**Purpose:** Understand the dataset, identify patterns, and discover key insights.

**Key Findings:**
- Dataset has 7,043 customers with 21 features
- Churn rate: 26.54% (imbalanced dataset)
- **Tenure** has the strongest negative correlation with churn (-0.352)
- **MonthlyCharges** positively correlates with churn (0.193)
- **Contract type** is a major churn indicator:
  - Month-to-month: ~43% churn rate
  - One year: ~11% churn rate
  - Two year: ~3% churn rate
- **Payment method** significantly impacts churn:
  - Electronic check: ~45% churn rate
  - Automatic payments: ~15-17% churn rate
- **Internet service type** matters:
  - Fiber optic: ~42% churn rate
  - DSL: ~19% churn rate
  - No internet: ~7% churn rate

**Actions Taken:**
- Identified missing values in `TotalCharges` (empty strings)
- Analyzed correlations between features and target
- Visualized churn patterns across categorical features

### 2. **02_feature_engineering.ipynb** - Feature Engineering
**Purpose:** Create new features to improve model performance and capture domain knowledge.

**Features Created (13 new features):**
1. **tenure_group** - Categorical tenure groups (0-1 year, 1-2 years, etc.)
2. **monthly_charges_bin** - Binned monthly charges (Low, Medium, High, Very High)
3. **avg_monthly_charges** - Average monthly charges (TotalCharges / tenure)
4. **charges_ratio** - Ratio of monthly to total charges
5. **service_count** - Count of services used by customer
6. **contract_payment** - Interaction feature (Contract × PaymentMethod)
7. **has_phone** - Binary flag for phone service
8. **has_internet** - Binary flag for internet service
9. **has_fiber** - Binary flag for fiber optic internet
10. **is_monthly_contract** - Binary flag for month-to-month contract
11. **is_electronic_check** - Binary flag for electronic check payment
12. **is_senior** - Binary flag for senior citizen
13. **no_protection** - Binary flag for customers without protection services

**Data Cleaning:**
- Converted `TotalCharges` from object to numeric (handled empty strings)
- Encoded target variable (Churn: Yes→1, No→0)
- Removed `customerID` (identifier, not a feature)

**Output:** Feature-engineered dataset saved to `data/features/data_with_features.csv`

### 3. **03_model_training.ipynb** - Baseline Model Training
**Purpose:** Train and compare baseline models to establish performance benchmarks.

**Models Trained:**
1. **LightGBM** (Gradient Boosting)
   - Parameters: n_estimators=100, learning_rate=0.1, max_depth=5
   - Class weights: balanced (to handle imbalanced data)
   - **Validation ROC AUC: 0.8432**

2. **CatBoost** (Gradient Boosting with categorical handling)
   - Parameters: iterations=100, learning_rate=0.1, depth=5
   - Auto class weights: Balanced
   - **Validation ROC AUC: 0.8390**

**Results:**
- **Best Model:** LightGBM (ROC AUC: 0.8432)
- **Test Set Performance:**
  - Accuracy: 0.7431
  - Precision: 0.5104
  - Recall: 0.7888
  - F1 Score: 0.6197
  - **ROC AUC: 0.8399**

**MLflow Integration:**
- All experiments logged to MLflow for tracking
- Models saved with metadata and metrics
- Easy comparison between model runs

### 4. **04_hyperparameter_tuning.ipynb** - Hyperparameter Optimization
**Purpose:** Optimize model hyperparameters using Optuna to improve performance.

**Optimization Approach:**
- **Framework:** Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Trials:** 50 trials per model
- **Objective:** Maximize ROC AUC on validation set
- **Early Stopping:** Enabled to prevent overfitting

**LightGBM Tuned Parameters:**
- num_leaves: 40
- max_depth: 4
- learning_rate: 0.295
- n_estimators: 338
- min_child_samples: 58
- subsample: 0.893
- colsample_bytree: 0.611
- reg_alpha: 0.004
- reg_lambda: 0.002
- scale_pos_weight: 2.732

**CatBoost Tuned Parameters:**
- iterations: 158
- depth: 5
- learning_rate: 0.078
- l2_leaf_reg: 1.029
- border_count: 56
- bagging_temperature: 1.184
- random_strength: 0.954
- scale_pos_weight: 3.829

**Results:**
- **Best Model:** LightGBM (Tuned)
  - **Validation ROC AUC: 0.8450** (improved from 0.8432)
  - **Test ROC AUC: 0.8439**
- **CatBoost (Tuned):**
  - Validation ROC AUC: 0.8400
  - Test ROC AUC: 0.8455

**Saved Artifacts:**
- Tuned model: `models/tuned_model.pkl`
- Best hyperparameters: `models/best_hyperparameters.pkl`
- Optuna studies: `models/optuna_studies.pkl`

### 5. **05_model_interpretability.ipynb** - Model Interpretation & Risk Scoring
**Purpose:** Understand model predictions and implement risk scoring for business use.

**SHAP Analysis:**
- **Top 10 Most Important Features:**
  1. Contract (mean absolute SHAP: 0.592)
  2. contract_payment (0.283)
  3. tenure (0.254)
  4. has_fiber (0.244)
  5. OnlineSecurity (0.235)
  6. TechSupport (0.174)
  7. charges_ratio (0.158)
  8. is_electronic_check (0.128)
  9. MonthlyCharges (0.113)
  10. avg_monthly_charges (0.083)

**Key Insights:**
- **Contract type** is the strongest predictor of churn
- **Tenure** significantly reduces churn probability (longer tenure = lower churn)
- **Fiber optic** customers have higher churn risk
- **Electronic check** payment method increases churn risk
- **Protection services** (OnlineSecurity, TechSupport) reduce churn

**Risk Scoring System:**
- **Risk Score Range:** 0-100 (based on churn probability)
- **Risk Levels:**
  - **LOW (0-39):** Unlikely to churn → Monthly check-in
  - **MEDIUM (40-69):** Some churn indicators → Weekly engagement
  - **HIGH (70-100):** Highly likely to churn → Immediate intervention (24-48 hours)

**Risk Scoring Module:**
- Implemented in `src/models/risk_scoring.py`
- Provides actionable risk categories with recommendations
- Supports batch and individual customer predictions
- Includes risk distribution analysis

## 🤖 Models Used

### 1. **LightGBM (Light Gradient Boosting Machine)**
- **Type:** Gradient Boosting Decision Tree
- **Advantages:**
  - Fast training and prediction
  - Handles categorical features efficiently
  - Built-in support for class imbalance
  - Excellent performance on tabular data
- **Final Performance:** ROC AUC = 0.8450 (validation), 0.8439 (test)

### 2. **CatBoost (Categorical Boosting)**
- **Type:** Gradient Boosting with advanced categorical handling
- **Advantages:**
  - Superior handling of categorical features
  - Built-in regularization to prevent overfitting
  - Robust to hyperparameter settings
  - Good performance with minimal tuning
- **Final Performance:** ROC AUC = 0.8400 (validation), 0.8455 (test)

**Model Selection:** LightGBM was selected as the final model based on validation ROC AUC, though both models performed similarly.

## 📊 Model Performance Summary

| Metric | LightGBM (Baseline) | LightGBM (Tuned) | CatBoost (Tuned) |
|--------|---------------------|------------------|------------------|
| **Validation ROC AUC** | 0.8432 | **0.8450** | 0.8400 |
| **Test ROC AUC** | 0.8399 | **0.8439** | 0.8455 |
| **Test Accuracy** | 0.7431 | - | - |
| **Test Precision** | 0.5104 | - | - |
| **Test Recall** | 0.7888 | - | - |
| **Test F1 Score** | 0.6197 | - | - |

## 🛠️ Technologies & Tools

- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, LightGBM, CatBoost
- **Hyperparameter Tuning:** Optuna
- **Model Interpretability:** SHAP
- **Experiment Tracking:** MLflow
- **Visualization:** matplotlib, seaborn, plotly
- **Configuration:** pydantic-settings
- **API Framework:** FastAPI (to be implemented)
- **Package Management:** uv (via pyproject.toml)

## 📈 What's Been Done So Far

✅ **Data Exploration & Analysis**
- Comprehensive EDA with visualizations
- Identified key churn drivers
- Analyzed feature correlations and distributions

✅ **Feature Engineering**
- Created 13 new features capturing domain knowledge
- Handled data quality issues (TotalCharges conversion)
- Prepared data for modeling

✅ **Model Development**
- Trained baseline models (LightGBM, CatBoost)
- Implemented class imbalance handling
- Achieved strong baseline performance (ROC AUC ~0.84)

✅ **Hyperparameter Optimization**
- Optimized both models using Optuna (50 trials each)
- Improved model performance through systematic tuning
- Saved best models and hyperparameters

✅ **Model Interpretability**
- SHAP analysis to understand feature importance
- Identified top churn risk factors
- Created visualizations for stakeholder communication

✅ **Risk Scoring System**
- Implemented risk scoring module (0-100 scale)
- Categorized customers into LOW/MEDIUM/HIGH risk
- Provided actionable recommendations per risk level

✅ **Experiment Tracking**
- Integrated MLflow for experiment management
- Tracked all model runs with parameters and metrics
- Enabled model versioning and comparison

## 🚀 Next Steps

### 1. **API Development** (Priority: High)
- [ ] Create FastAPI endpoints for model inference
- [ ] Implement batch prediction endpoint
- [ ] Add health check and model versioning endpoints
- [ ] Create API documentation (Swagger/OpenAPI)
- [ ] Add input validation and error handling

### 2. **Model Deployment** (Priority: High)
- [ ] Containerize application with Docker
- [ ] Set up CI/CD pipeline
- [ ] Deploy to cloud platform (AWS/GCP/Azure)
- [ ] Implement model serving infrastructure
- [ ] Set up load balancing and auto-scaling

### 3. **Model Monitoring** (Priority: Medium)
- [ ] Implement data drift detection
- [ ] Set up model performance monitoring
- [ ] Create alerting system for model degradation
- [ ] Track prediction distributions over time
- [ ] Monitor feature importance shifts

### 4. **Production Enhancements** (Priority: Medium)
- [ ] Implement A/B testing framework
- [ ] Add model retraining pipeline
- [ ] Create automated feature engineering pipeline
- [ ] Set up data validation checks
- [ ] Implement logging and observability

### 5. **Advanced Features** (Priority: Low)
- [ ] Implement ensemble methods (stacking/voting)
- [ ] Add time-series features (if temporal data available)
- [ ] Create customer segmentation based on churn risk
- [ ] Develop retention campaign recommendation system
- [ ] Build dashboard for business stakeholders

### 6. **Documentation & Testing** (Priority: Medium)
- [ ] Write comprehensive API documentation
- [ ] Add unit tests for core modules
- [ ] Create integration tests for API endpoints
- [ ] Document deployment procedures
- [ ] Create user guide for risk scoring system

## 🏃 Getting Started

### Prerequisites
- Python 3.10+
- uv (package manager) or pip

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd churn-prediction
```

2. **Install dependencies:**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e ".[all]"
```

3. **Run notebooks:**
```bash
jupyter notebook notebooks/
```

### Running the Risk Scoring Module

```python
from src.models.risk_scoring import create_risk_scorer
import pandas as pd

# Initialize risk scorer
risk_scorer = create_risk_scorer()

# Predict risk for a customer
customer_data = pd.DataFrame({
    # ... customer features ...
})

result = risk_scorer.predict_risk(customer_data, return_details=True)
print(f"Risk Level: {result.risk_level}")
print(f"Risk Score: {result.risk_score}")
print(f"Recommendation: {result.recommendation}")
```

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 📧 Contact

[Add contact information here]

---

**Note:** This project is actively under development. See the "Next Steps" section for planned improvements and features.

