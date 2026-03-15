# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lesson 07: Large Language Models (LLMs)
#
# In this lesson, we explore the evolution, architecture, and deployment of Large Language Models. We move beyond the standard transformer architecture to understand how modern models are specialized and how "Mixture of Experts" (MoE) designs enable massive scaling without equivalent increases in compute costs.

# %% [markdown]
# ## 1. LLM Foundations: From Base to Reasoning Models
#
# Large Language Models have evolved through several distinct stages of training and specialization.

# %% [markdown]
# ### 1.1. Foundational (Base) Models
# Foundational models are trained on massive corpora of text via **Next Token Prediction**. They learn the statistical structure of language, facts about the world, and basic reasoning capabilities.
# *   **Objective**: Predict the most likely next token.
# *   **Behavior**: They "complete" text. If you ask "What is the capital of France?", they might reply with "...and what is the capital of Germany?" because they've seen lists of questions in their training data.
# *   **Examples**: GPT-3 (original), Llama-2 (base), Mistral-7B-v0.1.

# %% [markdown]
# ### 1.2. Instruction-Tuned (Chat) Models
# To make models useful assistants, they undergo **Instruction Tuning** (SFT - Supervised Fine-Tuning) and often **RLHF** (Reinforcement Learning from Human Feedback).
# *   **Objective**: Follow natural language instructions and behave as helpful assistants.
# *   **Behavior**: Directly answers questions, summarizes text, or writes code based on a prompt.
# *   **Examples**: GPT-4o, Llama-3-Instruct, Claude 3.5 Sonnet.

# %% [markdown]
# ### 1.3. Reasoning (Chain-of-Thought) Models
# A recent shift involves models explicitly trained to "think" before they speak. They use internal **Chain-of-Thought (CoT)** reasoning to solve complex problems, often through Reinforcement Learning on reasoning traces.
# *   **Objective**: Solve complex math, logic, and coding problems with high accuracy.
# *   **Behavior**: Generates a hidden (or visible) reasoning trace before providing the final answer.
# *   **Examples**: OpenAI o1, DeepSeek-R1.

# %% [markdown]
# ## 2. Mixture of Experts (MoE)
#
# As models grew to hundreds of billions of parameters, "dense" transformers became prohibitively expensive to run. **Mixture of Experts (MoE)** is a sparse architecture that allows for massive parameter counts with relatively low inference costs.

# %% [markdown]
# ### 2.1. Sparse vs. Dense
# *   **Dense Model**: Every input token passes through *all* parameters in the model.
# *   **MoE Model**: For each token, a **Router** (Gating Network) selects only a few "Experts" (specialized Feed-Forward Networks) to process it.

# %% [markdown]
# ### 2.2. Mixtral 8x7B: A Popular Example
# Mixtral 8x7B has ~47B total parameters, but for each token, it only uses ~13B parameters (2 experts out of 8).
# *   **Structure**: The Attention layers are shared, but the Feed-Forward (FFN) layers are replaced by 8 independent experts.
# *   **Routing**: A router decides which 2 experts are best suited for the current token's context.
#
# <details>
# <summary><b>Diagram: MoE Routing Workflow</b></summary>
#
# ```mermaid
# graph TD
#     Input[Input Token] --> Attention[Shared Self-Attention]
#     Attention --> Router{Gating Router}
#     Router -->|Expert 1 weight| Exp1[Expert 1: Coding]
#     Router -->|Expert 2 weight| Exp2[Expert 2: Logic]
#     Router -.->|Disabled| Exp3[Expert 3: Creative]
#     Router -.->|Disabled| ExpN[Expert N]
#     Exp1 --> Agg[Weighted Sum / Softmax]
#     Exp2 --> Agg
#     Agg --> Output[Output Vector]
# ```
# </details>

# %% [markdown]
# ### 2.3. Qwen3-Coder-Next: Extreme MoE Efficiency
# One of the most efficient models for local agentic workflows is **Qwen3-Coder-Next**.
# *   **Parameters**: 80B total parameters, but only **3B active** parameters per token.
# *   **Architecture**: A sparse MoE with 512 experts (activating 10 + 1 shared).
# *   **Agentic Focus**: Specifically trained for long-horizon reasoning, tool use, and 256K (extendable to 1M) context windows, optimized to run on consumer hardware (e.g., Mac Studio or multi-GPU setups).

# %% [markdown]
# ### 2.4. Liquid Foundation Models (LFM): Beyond Transformers
# **Liquid Foundation Models** represent a shift away from traditional fixed-weight Transformers towards dynamical systems.
# *   **LFM-MoE**: Models like **LFM2-8B-A1B** use only 1.5B active parameters.
# *   **Efficiency**: They offer constant-memory complexity for long sequences and are designed for edge deployment (phones, laptops).

# %% [markdown]
# ## 3. Emergent Directions in LLMs
#
# The field is moving beyond "dense transformer + 16-bit precision" towards parallel generation and extreme efficiency.

# %% [markdown]
# ### 3.1. Diffusion Models for Text
# While Diffusion is famous for Image/Video (Stable Diffusion, Sora), it is now being applied to Text (e.g., **LLaDA**, **Mercury Coder**).
# *   **Parallel Generation**: Unlike autoregressive models (token-by-token), Diffusion models can generate an entire sequence of text in parallel.
# *   **Speed**: Can reach speeds of >1000 tokens/second.
#
# <details>
# <summary><b>Diagram: Autoregressive vs Diffusion Text Generation</b></summary>
#
# ```mermaid
# sequenceDiagram
#     participant AR as Autoregressive (Llama)
#     participant D as Diffusion (LLaDA)
#     
#     Note over AR: Token-by-Token
#     AR->>AR: Step 1: "The"
#     AR->>AR: Step 2: "The cat"
#     AR->>AR: Step 3: "The cat sat"
#     
#     Note over D: Parallel Denoising
#     D->>D: Step 1: [Noise] [Noise] [Noise]
#     D->>D: Step 2: [The] [Noise] [sat]
#     D->>D: Step 3: [The] [cat] [sat]
# ```
# </details>

# %% [markdown]
# ## 4. Inference Optimizations: The Engine Under the Hood
#
# To run these models efficiently, several key architectural optimizations are used in production servers like vLLM.

# %% [markdown]
# ### 4.1. The KV Cache: Eliminating Redundant Compute
# In autoregressive generation, each new token requires the model to "attend" to all previous tokens. Without a cache, we would recompute the hidden states (Keys and Values) for every single token in the context for every new step—an $O(N^2)$ process.
# *   **Mechanism**: We store the **Key** and **Value** vectors for all past tokens in memory (VRAM).
# *   **Benefit**: When generating the next token, we only compute K/V for the *new* token and concat it with the cache. This turns the generation step into an $O(N)$ operation.
# *   **Trade-off**: Memory capacity. A large context (e.g., 128k tokens) can require dozens of gigabytes of VRAM just for the KV Cache.
#
# <details>
# <summary><b>Diagram: KV Cache Mechanism</b></summary>
#
# ```mermaid
# graph TD
#     subgraph Step_1_The
#         T1[Token: The] --> KV1[Compute K1, V1]
#         KV1 --> Cache[(KV Cache)]
#     end
#     subgraph Step_2_The_cat
#         T2[Token: cat] --> KV2[Compute K2, V2]
#         Cache --> Attention{Attention Mechanism}
#         KV2 --> Attention
#         Attention --> Out2[Next Token: sat]
#         KV2 --> Cache
#     end
# ```
# </details>

# %% [markdown]
# ### 4.2. Flash Attention: IO-Awareness
# Traditional Attention has $O(N^2)$ memory complexity. **Flash Attention** optimizes how data is moved between High Bandwidth Memory (HBM) and SRAM on the GPU.
# *   **Tiling**: Breaks the attention matrix into blocks that fit into the GPU's fast cache (SRAM).
# *   **Zero Redundancy**: Computes Softmax without store/load operations of the large intermediate matrix ($N \times N$).

# %% [markdown]
# ### 4.3. Paged Attention: Memory Virtualization
# Serving multiple users simultaneously causes the **Key-Value (KV) Cache** to grow. Traditional systems pre-allocate large blocks of memory, leading to fragmentation.
# *   **Concept**: Inspired by OS Virtual Memory. It stores the KV cache in non-contiguous physical memory blocks.
# *   **Benefit**: Nearly 0% wasted memory, allowing for much larger batch sizes.
#
# <details>
# <summary><b>Diagram: Paged Attention Memory Layout</b></summary>
#
# ```mermaid
# graph LR
#     subgraph Logical_Space
#         S1[Seq 1: Block A]
#         S2[Seq 1: Block B]
#     end
#     subgraph Block_Table
#         BT1[Block A -> Physical 102]
#         BT2[Block B -> Physical 45]
#     end
#     subgraph Physical_VRAM
#         P102[Slot 102: Cache Data]
#         PXX[Slot 103: Empty]
#         P45[Slot 45: Cache Data]
#     end
#     S1 --> BT1 --> P102
#     S2 --> BT2 --> P45
# ```
# </details>

# %% [markdown]
# ### 4.4. Speculative Decoding: Draft & Verify
# Autoregressive generation is bottlenecked by Memory Bandwidth. **Speculative Decoding** uses a small model to "guess" tokens.
# 1.  **Draft**: A small model (e.g., Llama-7B) quickly generates 5 candidate tokens.
# 2.  **Verify**: The large model (e.g., Llama-405B) validates all 5 tokens in a *single* forward pass.
# 3.  **Acceptance**: If the large model says "Yes", we just gained 5 tokens for the cost of 1 large forward pass.
#
# <details>
# <summary><b>Diagram: Speculative Decoding Cycle</b></summary>
#
# ```mermaid
# graph LR
#     Small[Draft Model - 1B] -->|Guesses T1, T2, T3| Verify{Large Model - 70B}
#     Verify -->|Accepts All| Fast[3.5x Speedup]
#     Verify -->|Reject T3| Correct[Correct T3 & Reset]
# ```
# </details>

# %% [markdown]
# ### 4.1. Quantization: Shrinking the Model
# LLMs are typically trained in 16-bit precision (BF16/FP16). Quantization reduces this to 8-bit, 4-bit, or even 1.5-bit.
# *   **GGUF**: The standard format for `llama.cpp` and Ollama. Optimized for CPU/GPU inference.
# *   **MXFP**: The emerging standard for native hardware acceleration in next-gen AI chips.
# *   **AWQ/EXL2**: Formats optimized for high-speed GPU inference.
# *   **Impact**: A 7B parameter model in 4-bit (GGUF) fits into ~5GB of VRAM, making it runnable on most modern laptops.

# %% [markdown]
# ### 4.2. Local Runtimes
# *   **llama.cpp**: The foundational C++ implementation that started the local LLM revolution. Support for almost all hardware (Apple Silicon, CUDA, AVX).
# *   **Ollama**: A user-friendly wrapper around `llama.cpp` that provides a Docker-like CLI and a REST API.
# *   **vLLM**: A high-throughput serving engine for GPUs, using PagedAttention to handle many concurrent requests efficiently.

# %% [markdown]
# ## 5. Hands-on: Local Inference
#
# Running powerful models locally is now possible even on modest hardware. For this exercise, we suggest using **LFM2.5-1.2B-Instruct**, which offers high reasoning quality in a ~2.3 GB package.
#
# ### Step 1: Install Ollama
# Download from [ollama.com](https://ollama.com).
#
# ### Step 2: Run the Model
# Open your terminal and run:
# ```bash
# # Run the highly efficient Liquid Foundation Model 2.5
# ollama run tomng/lfm2.5-instruct
# ```
#
# ### Step 3: Experiment
# Try asking for a Romanian summary of a consumer protection topic, or a Python script to parse press releases. Observe the token generation speed (TPS).

# %% [markdown]
# ## 6. References and Further Reading
#
# ### Foundations & Reasoning
# *   [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/)
# *   [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
#
# ### Mixture of Experts (MoE)
# *   [Mixtral of Experts (Technical Blog)](https://mistral.ai/news/mixtral-of-experts/)
# *   [Qwen2.5-Coder: Empowering Everyone with Code (Technical Report)](https://arxiv.org/abs/2409.12186)
# *   [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
# *   [Liquid Foundation Models (Liquid AI Blog)](https://www.liquid.ai/blog/liquid-foundation-models)
#
# ### Emergent Directions (Diffusion & MXFP)
# *   [LLaDA: Large Language Diffusion with mAsking](https://arxiv.org/abs/2502.09992)
# *   [Mercury-Coder: High-speed coding via Diffusion (Inception Labs)](https://www.inceptionlabs.ai/)
# *   [OCP Microscaling Formats (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
#
# ### Inference Optimizations
# *   [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
# *   [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
# *   [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17115)
#
# ### Quantization
# *   [GGUF Specification (llama.cpp)](https://github.com/ggerganov/llama.cpp/blob/master/docs/gguf.md)
# *   [AWQ: Activation-aware Weight Quantization for LLM Compression](https://arxiv.org/abs/2306.00978)
