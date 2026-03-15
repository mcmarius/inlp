# Module 07: Large Language Models (LLMs)

This module explores the evolution, architecture, and deployment of Large Language Models (LLMs), moving beyond basic transformers into modern specialized designs.

## Content Overview

### 1. LLM Foundations
- **Base Models**: Trained on next-token prediction (e.g., Llama-2, Mistral).
- **Instruction-Tuned (Chat) Models**: Specialized via SFT and RLHF for assistant behavior (e.g., Llama-3-Instruct).
- **Reasoning Models**: Explicitly trained for complex problem-solving via Chain-of-Thought (e.g., DeepSeek-R1).

### 2. Mixture of Experts (MoE)
- Understanding sparse architectures (Mixtral 8x7B, Qwen3-Coder-Next).
- Scaling parameter counts while keeping inference costs low.
- Routing mechanisms and expert specialization.

### 3. Emergent Directions
- **Diffusion Models for Text**: Parallel generation vs. autoregressive token-by-token generation (e.g., LLaDA).
- **Liquid Foundation Models (LFM)**: Dynamical systems beyond fixed-weight Transformers.

### 4. Inference Optimizations
- **KV Cache**: Eliminating redundant computation.
- **Flash Attention**: IO-aware attention mechanisms.
- **Paged Attention**: Memory virtualization for high-throughput serving (vLLM).
- **Speculative Decoding**: Draft & Verify cycles for speedup.
- **Quantization**: Shrinking models to fit consumer hardware (GGUF, MXFP, AWQ).

## Hands-on Exercise
The provided notebook guides you through running powerful models locally using **Ollama** and experimenting with reasoning capabilities on modest hardware.

## Resources
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [vLLM Documentation](https://github.com/vllm-project/vllm)
