"""LIME (Local Interpretable Model-agnostic Explanations) wrapper for text classification."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from typing import List, Callable, Optional, Tuple, Any


class TextLimeExplainer:
    """Wrapper for LIME text explainer with visualization utilities."""
    
    def __init__(
        self,
        class_names: List[str],
        random_state: int = 42
    ):
        """
        Initialize LIME text explainer.
        
        Args:
            class_names: List of class names for classification
            random_state: Random seed for reproducibility
        """
        self.class_names = class_names
        self.explainer = LimeTextExplainer(
            class_names=class_names,
            random_state=random_state
        )
    
    def explain_instance(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Any:
        """
        Generate LIME explanation for a single text instance.
        
        Args:
            text: Input text to explain
            predict_fn: Function that takes list of texts and returns probabilities
            num_features: Number of features to include in explanation
            num_samples: Number of perturbed samples to generate
            
        Returns:
            LIME Explanation object
        """
        explanation = self.explainer.explain_instance(
            text,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )
        return explanation
    
    def get_top_features(
        self,
        explanation: Any,
        label: int = 1
    ) -> pd.DataFrame:
        """
        Extract top contributing features as DataFrame.
        
        Args:
            explanation: LIME Explanation object
            label: Class label to extract features for
            
        Returns:
            DataFrame with columns: feature, weight
        """
        features = explanation.as_list(label=label)
        
        df = pd.DataFrame(features, columns=['feature', 'weight'])
        df = df.sort_values('weight', key=abs, ascending=False)
        
        return df
    
    def visualize_explanation(
        self,
        explanation: Any,
        label: int = 1,
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Create matplotlib visualization of LIME explanation.
        
        Args:
            explanation: LIME Explanation object
            label: Class label to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        features_df = self.get_top_features(explanation, label=label)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if w > 0 else 'red' for w in features_df['weight']]
        
        ax.barh(
            range(len(features_df)),
            features_df['weight'],
            color=colors,
            alpha=0.7
        )
        ax.set_yticks(range(len(features_df)))
        ax.set_yticklabels(features_df['feature'])
        ax.set_xlabel('LIME Weight')
        ax.set_ylabel('Feature')
        ax.set_title(
            f'LIME Explanation for Class: {self.class_names[label]}',
            fontsize=14,
            fontweight='bold'
        )
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        return fig
    
    def show_in_notebook(
        self,
        explanation: Any,
        label: int = 1
    ):
        """
        Display HTML visualization in Jupyter notebook.
        
        Args:
            explanation: LIME Explanation object
            label: Class label to visualize
        """
        return explanation.show_in_notebook(labels=(label,))
