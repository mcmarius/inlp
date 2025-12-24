"""Feature importance extraction and visualization for traditional ML models."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def get_logistic_regression_coefficients(
    model: LogisticRegression,
    feature_names: List[str],
    class_idx: int = 1,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract and rank features by coefficient magnitude from Logistic Regression.
    
    Args:
        model: Trained LogisticRegression model
        feature_names: List of feature names corresponding to model coefficients
        class_idx: Class index for multi-class (0 for binary, or specific class for multi-class)
        top_n: Number of top features to return
        
    Returns:
        DataFrame with columns: feature, coefficient, abs_coefficient
    """
    if len(model.coef_.shape) == 1:
        # Binary classification with single coefficient vector
        coefficients = model.coef_
    elif model.coef_.shape[0] == 1:
        # Binary classification stored as (1, n_features)
        coefficients = model.coef_[0]
    else:
        # Multi-class classification
        coefficients = model.coef_[class_idx]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    })
    
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    return feature_importance.head(top_n)


def get_random_forest_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract feature importance from Random Forest.
    
    Args:
        model: Trained RandomForestClassifier model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with columns: feature, importance
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    return feature_importance.head(top_n)


def plot_top_features(
    feature_df: pd.DataFrame,
    title: str = "Top Features",
    value_col: str = "importance",
    figsize: Tuple[int, int] = (10, 8),
    color: str = "steelblue"
) -> plt.Figure:
    """
    Plot top features with their importance/coefficient values.
    
    Args:
        feature_df: DataFrame with 'feature' and value column
        title: Plot title
        value_col: Name of column containing importance values
        figsize: Figure size
        color: Bar color
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by absolute value if it's coefficients
    if 'coefficient' in feature_df.columns:
        plot_df = feature_df.copy()
        colors = ['red' if x < 0 else 'green' for x in plot_df['coefficient']]
        sns.barplot(
            data=plot_df,
            y='feature',
            x='coefficient',
            hue='feature',
            palette=colors,
            ax=ax,
            legend=False
        )
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Coefficient Value')
    else:
        sns.barplot(
            data=feature_df,
            y='feature',
            x=value_col,
            hue='feature',
            color=color,
            ax=ax,
            legend=False
        )
        ax.set_xlabel('Importance')
    
    ax.set_ylabel('Feature')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def compare_feature_importance(
    importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 15,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Compare feature importance across multiple models side-by-side.
    
    Args:
        importance_dict: Dictionary mapping model names to feature importance DataFrames
        top_n: Number of top features to show per model
        figsize: Figure size
        
    Returns:
        Matplotlib figure with subplots
    """
    n_models = len(importance_dict)
    fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=False)
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, feature_df) in enumerate(importance_dict.items()):
        ax = axes[idx]
        
        plot_df = feature_df.head(top_n).copy()
        
        # Determine if coefficients or importance
        if 'coefficient' in plot_df.columns:
            colors = ['red' if x < 0 else 'green' for x in plot_df['coefficient']]
            sns.barplot(
                data=plot_df,
                y='feature',
                x='coefficient',
                hue='feature',
                palette=colors,
                ax=ax,
                legend=False
            )
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.set_xlabel('Coefficient')
        else:
            sns.barplot(
                data=plot_df,
                y='feature',
                x='importance',
                hue='feature',
                palette='viridis',
                ax=ax,
                legend=False
            )
            ax.set_xlabel('Importance')
        
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature' if idx == 0 else '')
    
    plt.tight_layout()
    return fig
