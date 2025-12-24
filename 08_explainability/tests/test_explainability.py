"""Unit tests for explainability utilities."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from explainability.utils import (
    prepare_classification_task,
    train_baseline_models,
    evaluate_models,
    create_comparison_table
)
from explainability.feature_importance import (
    get_logistic_regression_coefficients,
    get_random_forest_importance,
)
from explainability.lime_explainer import TextLimeExplainer
from explainability.shap_explainer import TextShapExplainer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'content': [
            'amendă aplicată operator economic',
            'amendă sancțiune contravenție',
            'avertisment operator economic',
            'avertisment neconformitate',
            'amendă lei operator',
            'avertisment verificare',
        ] * 10,  # Repeat for more samples
        'lemmatized_content': [
            'amendă aplica operator economic',
            'amendă sancțiune contravenție',
            'avertisment operator economic',
            'avertisment neconformitate',
            'amendă lei operator',
            'avertisment verifica',
        ] * 10
    }
    return pd.DataFrame(data)


def test_prepare_classification_task(sample_data):
    """Test classification task preparation."""
    X_train, y_train, X_test, y_test, label_names = prepare_classification_task(
        sample_data,
        task_type="fine_vs_warning",
        test_size=0.2,
        random_state=42
    )
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert len(label_names) == 2
    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})


def test_train_baseline_models(sample_data):
    """Test baseline model training."""
    X_train, y_train, _, _, _ = prepare_classification_task(
        sample_data,
        task_type="fine_vs_warning",
        test_size=0.2,
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=50)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    models = train_baseline_models(X_train_vec, y_train, random_state=42)
    
    assert 'logistic' in models
    assert 'random_forest' in models
    assert 'svm' in models
    assert isinstance(models['logistic'], LogisticRegression)
    assert isinstance(models['random_forest'], RandomForestClassifier)


def test_evaluate_models(sample_data):
    """Test model evaluation."""
    X_train, y_train, X_test, y_test, _ = prepare_classification_task(
        sample_data,
        task_type="fine_vs_warning",
        test_size=0.2,
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=50)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    models = train_baseline_models(X_train_vec, y_train, random_state=42)
    results = evaluate_models(models, X_test_vec, y_test)
    
    assert isinstance(results, pd.DataFrame)
    assert 'Model' in results.columns
    assert 'Accuracy' in results.columns
    assert 'F1' in results.columns
    assert len(results) == 3  # Three models


def test_logistic_regression_coefficients(sample_data):
    """Test coefficient extraction from logistic regression."""
    X_train, y_train, _, _, _ = prepare_classification_task(
        sample_data,
        task_type="fine_vs_warning",
        test_size=0.2,
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=50)
    X_train_vec = vectorizer.fit_transform(X_train)
    feature_names = vectorizer.get_feature_names_out()
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    importance = get_logistic_regression_coefficients(
        model, feature_names, class_idx=1, top_n=10
    )
    
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'coefficient' in importance.columns
    assert 'abs_coefficient' in importance.columns
    assert len(importance) <= 10


def test_random_forest_importance(sample_data):
    """Test importance extraction from random forest."""
    X_train, y_train, _, _, _ = prepare_classification_task(
        sample_data,
        task_type="fine_vs_warning",
        test_size=0.2,
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=50)
    X_train_vec = vectorizer.fit_transform(X_train)
    feature_names = vectorizer.get_feature_names_out()
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train_vec, y_train)
    
    importance = get_random_forest_importance(model, feature_names, top_n=10)
    
    assert isinstance(importance, pd.DataFrame)
    assert 'feature' in importance.columns
    assert 'importance' in importance.columns
    assert len(importance) <= 10


def test_lime_explainer():
    """Test LIME explainer initialization."""
    explainer = TextLimeExplainer(class_names=['Class0', 'Class1'], random_state=42)
    
    assert explainer.class_names == ['Class0', 'Class1']
    assert explainer.explainer is not None


def test_shap_explainer(sample_data):
    """Test SHAP explainer initialization."""
    X_train, y_train, _, _, _ = prepare_classification_task(
        sample_data,
        task_type="fine_vs_warning",
        test_size=0.2,
        random_state=42
    )
    
    vectorizer = TfidfVectorizer(max_features=50)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    explainer = TextShapExplainer(model, X_train_vec, model_type="linear")
    
    assert explainer.model == model
    assert explainer.model_type == "linear"
    assert explainer.explainer is not None


def test_create_comparison_table():
    """Test comparison table creation."""
    df1 = pd.DataFrame({
        'feature': ['word1', 'word2', 'word3'],
        'importance': [0.5, 0.3, 0.2]
    })
    
    df2 = pd.DataFrame({
        'feature': ['word1', 'word3', 'word4'],
        'importance': [0.4, 0.3, 0.3]
    })
    
    comparison = create_comparison_table(
        {'Method1': df1, 'Method2': df2},
        top_n=5
    )
    
    assert isinstance(comparison, pd.DataFrame)
    assert 'Method1' in comparison.columns
    assert 'Method2' in comparison.columns
