Critic-Guided Action Memory: Architecture & Design

1. Overview

This document details the design of the Action Memory Module for the GR00T VLA model. The core goal is to bridge the gap between low-level control (actions) and high-level semantics (LLM reasoning) by introducing a structured, value-aware memory system.

Instead of storing raw frame-by-frame actions, we compress entire action trajectories into semantic "concepts" (latents), evaluate them using a Critic, and store them for long-horizon planning and retrieval.

2. Core Philosophy

The design is built on three key insights:

Trajectory as the Atomic Unit:
Single frames (Action, State) lack independent meaning. Only a sequence of actions (a trajectory) represents a coherent behavior (e.g., "pick up cup"). Therefore, memory must operate at the trajectory level.

Dual-Path Representation:

Raw Path (High-Freq Control): We preserve the raw, per-frame latents (B, T, D) for the Action Decoder (Diffusion Policy) to ensure kinematic fidelity and smoothness.

Semantic Path (High-Level Reasoning): We compress the trajectory into a single latent (B, 1, D) for the LLM and Memory Bank. This allows the LLM to reason about "intent" rather than "muscle movements."

Critic-Guided Optimization:
By training a lightweight Critic on top of the LLM-processed latent, we provide a task-aligned signal (e.g., success/failure, stability). This gradients flows back to the encoder, forcing it to learn features relevant to task success.

3. Architecture Components

A. Trajectory Encoder (Causal Transformer)

We replace simple pooling with a Transformer-based Encoder to capture temporal dependencies and causal structure.

Input: Sequence of raw action latents (B, T, D_action).

Mechanism:

Append [CLS] Token: A learnable token is prepended to the sequence [CLS, t1, t2, ..., tT].

Positional Embedding: We apply a learnable positional embedding to the entire sequence so the model distinguishes "start" from "end."

Self-Attention: A Transformer Encoder processes the sequence. The [CLS] token aggregates information from all time steps via attention, automatically focusing on key moments (e.g., contact, stopping).

Output: The processed [CLS] token (B, 1, D_hidden) becomes the Trajectory Latent.

B. LLM Injection & Projection

Projector: A linear layer projects the Trajectory Latent to the LLM's embedding dimension (D_hidden -> D_llm).

LLM Pass: The projected latent is fed into the frozen LLM backbone (e.g., Eagle/Llama). This embeds the action concept into a rich, pre-trained semantic space.

C. Critic Head

Input: The LLM-processed semantic latent.

Architecture: A lightweight MLP (Linear -> LayerNorm -> SiLU -> Linear).

Output: A scalar value v representing the quality of the trajectory (e.g., predicted success probability or stability score).

Loss: Trained via MSE Loss against ground-truth outcome signals.

D. Memory Bank

A dynamic storage system that saves tuples of:

Semantic Latent: For retrieval and reasoning.

Raw Latent: For replaying or decoding into actions.

Critic Value: For filtering and prioritizing high-quality memories.

4. Implementation Details

File Structure

gr00t/model/action_memory/memory_module.py: Contains ActionMemory (Main Module), TrajectoryCompressor (Transformer Encoder), and CriticHead.

gr00t/model/gr00t_n1.py: Integration into the main VLA model forward pass via hook injection.

Data Flow

Input: Actions (B, T, D)

Encode: ActionEncoder (Existing) -> Raw Latents (B, T, D)

Compress: TrajectoryCompressor -> [CLS] + PosEmb -> Transformer -> Trajectory Latent (B, 1, D)

Reason: Projector -> LLM Backbone -> Semantic Latent

Evaluate: Semantic Latent -> CriticHead -> Value

Output: Dict containing critic_value, semantic_latent, and raw_latents.

Training Objective

The module is trained end-to-end (or staged) with the following objective:


$$L_{total} = L_{action\_diffusion} + \lambda \cdot L_{critic}(V_{pred}, V_{target})$$

$V_{target}$ is derived from environment feedback (e.g., -distance_to_goal, success_flag).

5. Future Considerations

Retrieval Mechanism: Currently a placeholder. Future work involves implementing Vector Search (e.g., Cosine Similarity) on the Semantic Latents to retrieve relevant past experiences.

Contrastive Learning: We could add an auxiliary loss to pull the Semantic Latent closer to the text embedding of the task description (e.g., "pick up red block") to enforce language alignment.