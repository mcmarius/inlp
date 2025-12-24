# Module 09: Feature Importance, Explainability, and Interpretability

In this module, we explore **how and why** NLP models make predictions. We build on previous modules (sparse/dense representations, transformers) to demonstrate explainability techniques from traditional ML through transformers, using the ANPC press releases dataset.

## Objectives

- Understand **feature importance** in traditional ML models (Logistic Regression, Random Forest)
- Apply **LIME** (Local Interpretable Model-agnostic Explanations) to text classification
- Use **SHAP** (SHapley Additive exPlanations) for model explanations
- Visualize **transformer attention** patterns to understand what models focus on
- Compare explainability methods across different model types
- Ground all insights in **real data** from the ANPC corpus

## Contents

- `explainability/`: Core implementation of explainability utilities.
    - `feature_importance.py`: Traditional ML feature importance extraction and visualization.
    - `lime_explainer.py`: LIME wrapper for text classification.
    - `shap_explainer.py`: SHAP wrapper for model explanations.
    - `attention_viz.py`: Transformer attention visualization utilities.
    - `utils.py`: Shared utilities for classification tasks and model training.
- `notebooks/`: Interactive walkthrough.
    - `09_explainability_and_interpretability.py`: Comprehensive notebook covering all techniques.
- `tests/`: Unit tests for explainability utilities.

## How to use

1. Generate the notebook from the JupyText script:
   ```bash
   uv run jupytext --update --to ipynb notebooks/09_explainability_and_interpretability.py
   ```
2. Explore `notebooks/09_explainability_and_interpretability.ipynb` to see the results.
3. Run the unit tests:
   ```bash
   uv run pytest tests/
   ```

## Key Techniques

### Traditional ML Feature Importance
- **Logistic Regression Coefficients**: Direct interpretation of feature weights
- **Random Forest Importance**: Gini importance and permutation importance
- **Comparison**: Understanding when different methods agree/disagree

### LIME (Local Interpretable Model-agnostic Explanations)
- **Local Approximation**: Explains individual predictions with interpretable models
- **Model-Agnostic**: Works with any classifier (Logistic Regression, Random Forest, Transformers)
- **Text-Specific**: Perturbs text by removing words to measure impact

### SHAP (SHapley Additive exPlanations)
- **Game Theory**: Based on Shapley values from cooperative game theory
- **Additive Feature Attribution**: Consistent and locally accurate explanations
- **Multiple Visualizations**: Waterfall, summary, and force plots

### Transformer Attention Visualization
- **Self-Attention Mechanism**: Understanding what tokens attend to each other
- **Multi-Head Attention**: Different heads capture different patterns
- **Layer-Wise Analysis**: How attention evolves through the network

## Visualizations

- **Feature Importance Plots**: Bar charts comparing top features across models
- **LIME Explanations**: Highlighted text showing positive/negative contributions
- **SHAP Waterfall Plots**: Step-by-step breakdown of prediction
- **Attention Heatmaps**: Token-to-token attention patterns
- **Comparative Analysis**: Side-by-side method comparisons

> [!NOTE]
> **Data Grounding**: All observations and insights in this module are derived from actual execution on the ANPC dataset. No hallucinated examples or statistics.
