"""
Risk Scoring Module for Churn Prediction

This module provides functionality to calculate customer churn risk scores
based on model predictions and categorize customers into risk levels.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from src.config import settings


@dataclass
class RiskScoreResult:
    """Container for risk score calculation results."""
    churn_probability: float
    risk_score: int
    risk_level: str
    risk_category: str
    recommendation: str


class RiskScorer:
    """
    Risk scoring system for churn prediction.
    
    Converts model predictions into actionable risk scores (0-100)
    and categorizes customers into risk levels.
    """
    
    # Risk level thresholds
    RISK_THRESHOLDS = {
        'LOW': (0, 39),
        'MEDIUM': (40, 69),
        'HIGH': (70, 100)
    }
    
    # Risk category descriptions
    RISK_CATEGORIES = {
        'LOW': {
            'name': 'Low Risk',
            'description': 'Customer is unlikely to churn',
            'action': 'Maintain current service level',
            'priority': 'Low',
            'intervention_timing': 'Monthly check-in'
        },
        'MEDIUM': {
            'name': 'Medium Risk',
            'description': 'Customer shows some churn indicators',
            'action': 'Engage with retention campaigns',
            'priority': 'Medium',
            'intervention_timing': 'Weekly engagement'
        },
        'HIGH': {
            'name': 'High Risk',
            'description': 'Customer is highly likely to churn',
            'action': 'Immediate retention intervention required',
            'priority': 'High',
            'intervention_timing': 'Within 24-48 hours'
        }
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the RiskScorer.
        
        Args:
            model_path: Path to the trained model. If None, uses default from settings.
        """
        if model_path is None:
            model_path = settings.MODELS_PATH / "tuned_model.pkl"
        
        self.model_path = Path(model_path)
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the model has been trained and saved."
            )
        return joblib.load(self.model_path)
    
    def calculate_risk_score(
        self, 
        churn_probability: float
    ) -> int:
        """
        Convert churn probability to risk score (0-100).
        
        Args:
            churn_probability: Probability of churn (0.0 to 1.0)
            
        Returns:
            Risk score as integer from 0 to 100
        """
        # Ensure probability is in valid range
        churn_probability = np.clip(churn_probability, 0.0, 1.0)
        
        # Convert to risk score (0-100 scale)
        risk_score = int(round(churn_probability * 100))
        
        return risk_score
    
    def categorize_risk(self, risk_score: int) -> str:
        """
        Categorize risk score into risk level.
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Risk level: 'LOW', 'MEDIUM', or 'HIGH'
        """
        risk_score = np.clip(risk_score, 0, 100)
        
        if risk_score <= self.RISK_THRESHOLDS['LOW'][1]:
            return 'LOW'
        elif risk_score <= self.RISK_THRESHOLDS['MEDIUM'][1]:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def get_risk_category_info(self, risk_level: str) -> Dict:
        """
        Get detailed information about a risk category.
        
        Args:
            risk_level: Risk level ('LOW', 'MEDIUM', or 'HIGH')
            
        Returns:
            Dictionary with risk category information
        """
        return self.RISK_CATEGORIES.get(risk_level, {})
    
    def predict_risk(
        self, 
        customer_data: Union[pd.DataFrame, pd.Series, np.ndarray, Dict],
        return_details: bool = True
    ) -> Union[RiskScoreResult, Dict]:
        """
        Predict churn risk for a single customer or multiple customers.
        
        Args:
            customer_data: Customer feature data (DataFrame, Series, array, or dict)
            return_details: If True, return detailed RiskScoreResult, else return dict
            
        Returns:
            RiskScoreResult or dict with risk information
        """
        # Convert to DataFrame if needed
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        elif isinstance(customer_data, pd.Series):
            customer_df = customer_data.to_frame().T
        elif isinstance(customer_data, np.ndarray):
            # Assume it's a single row
            if customer_data.ndim == 1:
                customer_df = pd.DataFrame([customer_data])
            else:
                customer_df = pd.DataFrame(customer_data)
        else:
            customer_df = customer_data.copy()
        
        # Ensure DataFrame has correct shape
        if customer_df.shape[0] > 1:
            # Multiple customers - process all
            return self.predict_risk_batch(customer_df, return_details)
        
        # Single customer prediction
        churn_proba = self.model.predict_proba(customer_df)[0][1]
        risk_score = self.calculate_risk_score(churn_proba)
        risk_level = self.categorize_risk(risk_score)
        category_info = self.get_risk_category_info(risk_level)
        
        result = RiskScoreResult(
            churn_probability=float(churn_proba),
            risk_score=risk_score,
            risk_level=risk_level,
            risk_category=category_info.get('name', risk_level),
            recommendation=category_info.get('action', 'No recommendation')
        )
        
        if return_details:
            return result
        else:
            return {
                'churn_probability': float(churn_proba),
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_category': category_info.get('name', risk_level),
                'description': category_info.get('description', ''),
                'recommendation': category_info.get('action', ''),
                'priority': category_info.get('priority', ''),
                'intervention_timing': category_info.get('intervention_timing', '')
            }
    
    def predict_risk_batch(
        self,
        customer_data: pd.DataFrame,
        return_details: bool = False
    ) -> pd.DataFrame:
        """
        Predict risk scores for multiple customers.
        
        Args:
            customer_data: DataFrame with customer features
            return_details: If True, include detailed information
            
        Returns:
            DataFrame with risk scores and categories
        """
        # Get predictions
        churn_probas = self.model.predict_proba(customer_data)[:, 1]
        
        # Calculate risk scores
        risk_scores = [self.calculate_risk_score(prob) for prob in churn_probas]
        risk_levels = [self.categorize_risk(score) for score in risk_scores]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'churn_probability': churn_probas,
            'risk_score': risk_scores,
            'risk_level': risk_levels
        })
        
        if return_details:
            # Add category information
            results['risk_category'] = results['risk_level'].map(
                lambda x: self.RISK_CATEGORIES[x]['name']
            )
            results['recommendation'] = results['risk_level'].map(
                lambda x: self.RISK_CATEGORIES[x]['action']
            )
            results['priority'] = results['risk_level'].map(
                lambda x: self.RISK_CATEGORIES[x]['priority']
            )
            results['intervention_timing'] = results['risk_level'].map(
                lambda x: self.RISK_CATEGORIES[x]['intervention_timing']
            )
        
        return results
    
    def get_risk_distribution(
        self,
        customer_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get distribution of risk levels in a dataset.
        
        Args:
            customer_data: DataFrame with customer features
            
        Returns:
            DataFrame with risk level distribution statistics
        """
        results = self.predict_risk_batch(customer_data, return_details=False)
        
        distribution = results['risk_level'].value_counts().to_frame('count')
        distribution['percentage'] = (
            distribution['count'] / len(results) * 100
        ).round(2)
        
        # Add risk score statistics
        stats = results.groupby('risk_level')['risk_score'].agg([
            'mean', 'min', 'max', 'std'
        ]).round(2)
        
        distribution = distribution.join(stats)
        distribution = distribution.sort_index()
        
        return distribution
    
    def get_high_risk_customers(
        self,
        customer_data: pd.DataFrame,
        threshold: int = 70,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Identify high-risk customers.
        
        Args:
            customer_data: DataFrame with customer features
            threshold: Risk score threshold (default: 70)
            top_n: Return top N customers by risk score (optional)
            
        Returns:
            DataFrame with high-risk customers sorted by risk score
        """
        results = self.predict_risk_batch(customer_data, return_details=True)
        
        # Filter by threshold
        high_risk = results[results['risk_score'] >= threshold].copy()
        
        # Sort by risk score (descending)
        high_risk = high_risk.sort_values('risk_score', ascending=False)
        
        # Return top N if specified
        if top_n is not None:
            high_risk = high_risk.head(top_n)
        
        return high_risk


def create_risk_scorer(model_path: Optional[Path] = None) -> RiskScorer:
    """
    Factory function to create a RiskScorer instance.
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        RiskScorer instance
    """
    return RiskScorer(model_path=model_path)

