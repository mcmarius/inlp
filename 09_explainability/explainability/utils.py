"""Utilities for preparing classification tasks and training baseline models."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple, List, Any


def prepare_classification_task(
    df: pd.DataFrame,
    task_type: str = "category",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Prepare a classification task from the ANPC dataset.
    
    Args:
        df: DataFrame with ANPC articles
        task_type: Type of classification task:
            - "category": Predict article category (if available)
            - "fine_vs_warning": Binary classification of articles with fines vs warnings
            - "high_fine": Binary classification of high vs low fine amounts
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test, label_names
    """
    if task_type == "fine_vs_warning":
        # Binary: articles mentioning fines vs warnings
        df = df.copy()
        df['has_fine'] = df['content'].str.contains(r'amen[dz]', case=False, regex=True)
        df['has_warning'] = df['content'].str.contains(r'avertis', case=False, regex=True)
        
        # Keep only articles with clear signal
        df_filtered = df[(df['has_fine'] | df['has_warning']) & ~(df['has_fine'] & df['has_warning'])]
        
        X = df_filtered['lemmatized_content'].tolist()
        y = df_filtered['has_fine'].astype(int).values
        label_names = ['Warning', 'Fine']
        
    elif task_type == "high_fine":
        # Binary: high fine amounts vs low
        df = df.copy()
        # Extract fine amounts (simplified - looks for "milioane" or large numbers)
        df['has_high_fine'] = df['content'].str.contains(r'milioane?\s+(?:de\s+)?lei', case=False, regex=True)
        df['has_fine'] = df['content'].str.contains(r'amen[dz]', case=False, regex=True)
        
        df_filtered = df[df['has_fine']]
        
        X = df_filtered['lemmatized_content'].tolist()
        y = df_filtered['has_high_fine'].astype(int).values
        label_names = ['Low Fine', 'High Fine']
        
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, y_train, X_test, y_test, label_names


def train_baseline_models(
    X_train_vec: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train baseline classification models.
    
    Args:
        X_train_vec: Vectorized training features (e.g., TF-IDF)
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Logistic Regression
    models['logistic'] = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        class_weight='balanced'
    )
    models['logistic'].fit(X_train_vec, y_train)
    
    # Random Forest
    models['random_forest'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        class_weight='balanced'
    )
    models['random_forest'].fit(X_train_vec, y_train)
    
    # SVM (optional, for comparison)
    models['svm'] = SVC(
        kernel='linear',
        probability=True,
        random_state=random_state,
        class_weight='balanced'
    )
    models['svm'].fit(X_train_vec, y_train)
    
    return models


def evaluate_models(
    models: Dict[str, Any],
    X_test_vec: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate multiple models and return metrics.
    
    Args:
        models: Dictionary of trained models
        X_test_vec: Vectorized test features
        y_test: Test labels
        
    Returns:
        DataFrame with evaluation metrics
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test_vec)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })
    
    return pd.DataFrame(results)


def create_comparison_table(
    feature_importance_dict: Dict[str, pd.DataFrame],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Create a comparison table of top features across different methods.
    
    Args:
        feature_importance_dict: Dictionary mapping method names to DataFrames with
                                 'feature' and 'importance' columns
        top_n: Number of top features to include
        
    Returns:
        DataFrame with features as rows and methods as columns
    """
    all_features = set()
    for df in feature_importance_dict.values():
        all_features.update(df.head(top_n)['feature'].tolist())
    
    comparison = pd.DataFrame(index=sorted(all_features))
    
    for method_name, df in feature_importance_dict.items():
        feature_scores = dict(zip(df['feature'], df['importance']))
        comparison[method_name] = comparison.index.map(lambda f: feature_scores.get(f, 0))
    
    # Sort by sum of importance across methods
    comparison['total'] = comparison.sum(axis=1)
    comparison = comparison.sort_values('total', ascending=False).drop('total', axis=1)
    
    return comparison.head(top_n)
