"""SHAP (SHapley Additive exPlanations) wrapper for model explanations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Optional, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class TextShapExplainer:
    """Wrapper for SHAP explainers with visualization utilities."""
    
    def __init__(
        self,
        model: Any,
        X_train: np.ndarray,
        model_type: str = "linear"
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model (LogisticRegression, RandomForest, etc.)
            X_train: Training data (vectorized) for background distribution
            model_type: Type of model ("linear", "tree", or "kernel")
        """
        self.model = model
        self.model_type = model_type
        
        if model_type == "linear":
            self.explainer = shap.LinearExplainer(model, X_train)
        elif model_type == "tree":
            self.explainer = shap.TreeExplainer(model)
        else:
            # Kernel explainer is model-agnostic but slower
            self.explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_train, 100)  # Use sample for efficiency
            )
    
    def explain_predictions(
        self,
        X_test: np.ndarray
    ) -> np.ndarray:
        """
        Generate SHAP values for test predictions.
        
        Args:
            X_test: Test data (vectorized)
            
        Returns:
            SHAP values array
        """
        # Convert sparse matrix to dense if needed (SHAP doesn't handle sparse well)
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        
        shap_values = self.explainer.shap_values(X_test)
        return shap_values
    
    def plot_waterfall(
        self,
        shap_values: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        instance_idx: int = 0,
        class_idx: int = 1,
        max_display: int = 10
    ) -> plt.Figure:
        """
        Create waterfall plot for a single prediction.
        
        Args:
            shap_values: SHAP values from explain_predictions
            X_test: Test data
            feature_names: List of feature names
            instance_idx: Index of instance to explain
            class_idx: Class index for multi-class models
            max_display: Maximum number of features to display
            
        Returns:
            Matplotlib figure
        """
        # Handle multi-class vs binary
        if isinstance(shap_values, list):
            # Multi-class: list of arrays, one per class
            values = shap_values[class_idx][instance_idx]
            base_value = self.explainer.expected_value[class_idx] if isinstance(
                self.explainer.expected_value, (list, np.ndarray)
            ) else self.explainer.expected_value
        elif len(shap_values.shape) == 3:
            # Multi-class: (n_samples, n_features, n_classes)
            values = shap_values[instance_idx, :, class_idx]
            base_value = self.explainer.expected_value[class_idx] if isinstance(
                self.explainer.expected_value, (list, np.ndarray)
            ) else self.explainer.expected_value
        else:
            # Binary: (n_samples, n_features)
            values = shap_values[instance_idx]
            base_value = self.explainer.expected_value if not isinstance(
                self.explainer.expected_value, (list, np.ndarray)
            ) else self.explainer.expected_value[0]
        
        # Convert sparse matrix to dense if needed
        if hasattr(X_test, 'toarray'):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = X_test
        
        # Create explanation object
        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=X_test_dense[instance_idx],
            feature_names=feature_names
        )
        
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        
        return fig
    
    def plot_summary(
        self,
        shap_values: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        class_idx: Optional[int] = None,
        max_display: int = 20
    ) -> plt.Figure:
        """
        Create summary plot across multiple predictions.
        
        Args:
            shap_values: SHAP values from explain_predictions
            X_test: Test data
            feature_names: List of feature names
            class_idx: Class index for multi-class (None for binary)
            max_display: Maximum number of features to display
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(10, 8))
        
        # Handle multi-class vs binary
        if isinstance(shap_values, list):
            # Multi-class: list of arrays
            values = shap_values[class_idx] if class_idx is not None else shap_values[0]
        elif len(shap_values.shape) == 3:
            # Multi-class: (n_samples, n_features, n_classes)
            values = shap_values[:, :, class_idx] if class_idx is not None else shap_values[:, :, 0]
        else:
            # Binary: (n_samples, n_features)
            values = shap_values
        
        # Convert sparse matrix to dense if needed
        if hasattr(X_test, 'toarray'):
            X_test_dense = X_test.toarray()
        else:
            X_test_dense = X_test
        
        shap.summary_plot(
            values,
            X_test_dense,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        plt.tight_layout()
        return fig
    
    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        class_idx: Optional[int] = None,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Extract top features by mean absolute SHAP value.
        
        Args:
            shap_values: SHAP values from explain_predictions
            feature_names: List of feature names
            class_idx: Class index for multi-class (None for binary)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with columns: feature, mean_abs_shap
        """
        # Handle multi-class vs binary
        if isinstance(shap_values, list):
            # Multi-class: list of arrays
            values = shap_values[class_idx] if class_idx is not None else shap_values[0]
        elif len(shap_values.shape) == 3:
            # Multi-class: (n_samples, n_features, n_classes)
            values = shap_values[:, :, class_idx] if class_idx is not None else shap_values[:, :, 0]
        else:
            # Binary: (n_samples, n_features)
            values = shap_values
        
        mean_abs_shap = np.abs(values).mean(axis=0)
        
        # Ensure mean_abs_shap is 1D
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        })
        
        feature_importance = feature_importance.sort_values(
            'mean_abs_shap',
            ascending=False
        )
        
        return feature_importance.head(top_n)
