"""Transformer attention visualization utilities."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional, Dict
from transformers import AutoTokenizer, AutoModel


def extract_attention_weights(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    text: str,
    device: str = "cpu"
) -> Tuple[List[str], torch.Tensor]:
    """
    Extract attention weights from a transformer model.
    
    Args:
        model: Transformer model (e.g., DistilBERT)
        tokenizer: Corresponding tokenizer
        text: Input text
        device: Device to run on ("cpu" or "cuda")
        
    Returns:
        Tuple of (tokens, attention_weights)
        - tokens: List of token strings
        - attention_weights: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
    """
    model.eval()
    model.to(device)
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention weights
    # outputs.attentions is a tuple of (num_layers,) each with shape (batch, heads, seq, seq)
    attention = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
    attention = attention.squeeze(1)  # Remove batch dimension: (layers, heads, seq, seq)
    
    return tokens, attention


def plot_attention_heatmap(
    tokens: List[str],
    attention: torch.Tensor,
    layer_idx: int = -1,
    head_idx: int = 0,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Visualize attention as a heatmap.
    
    Args:
        tokens: List of token strings
        attention: Attention tensor (layers, heads, seq, seq)
        layer_idx: Layer index to visualize (-1 for last layer)
        head_idx: Attention head index
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Extract specific layer and head
    attn_matrix = attention[layer_idx, head_idx].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': 'Attention Weight'}
    )
    
    ax.set_title(
        f'Attention Heatmap - Layer {layer_idx}, Head {head_idx}',
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig


def aggregate_attention(
    attention: torch.Tensor,
    method: str = "mean"
) -> torch.Tensor:
    """
    Aggregate attention across layers and/or heads.
    
    Args:
        attention: Attention tensor (layers, heads, seq, seq)
        method: Aggregation method ("mean", "max", "last_layer")
        
    Returns:
        Aggregated attention tensor (seq, seq)
    """
    if method == "mean":
        # Average across all layers and heads
        return attention.mean(dim=(0, 1))
    elif method == "max":
        # Max across all layers and heads
        return attention.max(dim=0)[0].max(dim=0)[0]
    elif method == "last_layer":
        # Average across heads in last layer
        return attention[-1].mean(dim=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def find_important_tokens(
    tokens: List[str],
    attention: torch.Tensor,
    aggregation: str = "mean",
    top_n: int = 10
) -> pd.DataFrame:
    """
    Identify tokens with highest attention scores.
    
    Args:
        tokens: List of token strings
        attention: Attention tensor (layers, heads, seq, seq)
        aggregation: How to aggregate attention ("mean", "max", "last_layer")
        top_n: Number of top tokens to return
        
    Returns:
        DataFrame with columns: token, attention_score
    """
    # Aggregate attention
    agg_attention = aggregate_attention(attention, method=aggregation)
    
    # Sum attention received by each token (column-wise sum)
    attention_received = agg_attention.sum(dim=0).cpu().numpy()
    
    # Create DataFrame
    token_importance = pd.DataFrame({
        'token': tokens,
        'attention_score': attention_received
    })
    
    # Remove special tokens
    token_importance = token_importance[
        ~token_importance['token'].isin(['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>'])
    ]
    
    token_importance = token_importance.sort_values(
        'attention_score',
        ascending=False
    )
    
    return token_importance.head(top_n)


def plot_attention_flow(
    tokens: List[str],
    attention: torch.Tensor,
    source_token_idx: int,
    layer_idx: int = -1,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Visualize attention flow from a specific source token.
    
    Args:
        tokens: List of token strings
        attention: Attention tensor (layers, heads, seq, seq)
        source_token_idx: Index of source token
        layer_idx: Layer index (-1 for last layer)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Average across heads for the specified layer
    layer_attention = attention[layer_idx].mean(dim=0).cpu().numpy()
    
    # Get attention from source token to all others
    attention_from_source = layer_attention[source_token_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(range(len(tokens)), attention_from_source, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_xlabel('Target Tokens')
    ax.set_ylabel('Attention Weight')
    ax.set_title(
        f'Attention Flow from "{tokens[source_token_idx]}" (Layer {layer_idx})',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    return fig
