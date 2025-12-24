"""Explainability utilities for NLP models."""

from .feature_importance import (
    get_logistic_regression_coefficients,
    get_random_forest_importance,
    plot_top_features,
    compare_feature_importance,
)
from .lime_explainer import TextLimeExplainer
from .shap_explainer import TextShapExplainer
from .attention_viz import (
    extract_attention_weights,
    plot_attention_heatmap,
    aggregate_attention,
    find_important_tokens,
)
from .utils import (
    prepare_classification_task,
    train_baseline_models,
    evaluate_models,
    create_comparison_table,
)

__all__ = [
    "get_logistic_regression_coefficients",
    "get_random_forest_importance",
    "plot_top_features",
    "compare_feature_importance",
    "TextLimeExplainer",
    "TextShapExplainer",
    "extract_attention_weights",
    "plot_attention_heatmap",
    "aggregate_attention",
    "find_important_tokens",
    "prepare_classification_task",
    "train_baseline_models",
    "evaluate_models",
    "create_comparison_table",
]
