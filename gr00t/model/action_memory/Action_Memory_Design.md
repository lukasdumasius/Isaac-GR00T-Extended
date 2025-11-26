# **Critic-Guided Action Memory: Architecture & Design**

## **1. Overview**

This document describes the design of the **Action Memory Module** for the **GR00T VLA** model.
The goal is to bridge the gap between:

* **Low-level control** (continuous actions)
* **High-level semantics** (LLM reasoning)

by introducing a **structured, value-aware memory system**.

Rather than storing raw frame-by-frame actions, we:

1. **Encode entire trajectories** into compact semantic latents
2. **Evaluate** them with a lightweight value **Critic**
3. **Store** only the semantic intent and compressed control blueprint
4. **Retrieve** relevant memories dynamically using **Gated Cross-Attention**

---

## **2. Core Philosophy**

The design is guided by three foundational insights.

---

### **2.1 Trajectory as the Atomic Unit**

A single action or frame lacks coherent meaning.
A *trajectory* represents interpretable concepts such as:

* “pick up the cup”
* “close the drawer”
* “push object to target”

Thus, memory must operate at the **trajectory** level.

---

### **2.2 Dual-Path Representation**

We maintain two parallel pathways:

#### **Raw Path → High-Frequency Control**

* Retains (B, T, D) per-step latents
* Ensures smoothness and kinematic fidelity for Diffusion Policies
* **Note**: Raw latents are used for immediate decoding but NOT stored in the long-term Memory Bank to save space.

#### **Semantic Path → High-Level Reasoning**

* Compress trajectory → single latent (B, 1, D)
* Used for LLM reasoning + memory indexing (Key)
* Represents “intent,” not motor signals

---

### **2.3 Critic-Guided Optimization**

A lightweight **Critic** evaluates trajectory quality (success probability, stability, etc.).

The Critic’s gradients:

* Flow back into the trajectory encoder
* Encourage useful, task-relevant features
* Shape the semantic latent space

---

## **3. Architecture Components**

---

### **3.1 Trajectory Encoder (Causal Transformer)**

Encodes raw action sequences into a single semantic trajectory latent.

#### **Input**

```
Raw Action Latents: (B, T, D_action)
```

#### **Mechanism**

* **[CLS] Token** is prepended
* **Learnable Positional Embedding**
* **Transformer Encoder** with causal structure
* **[CLS] token output** = trajectory summary

#### **Output**

```
Trajectory Latent: (B, 1, D_hidden)
```

---

### **3.2 LLM Injection & Projection**

* **Projection Layer**
  `D_hidden → D_llm`

* **LLM Backbone (Frozen)**

  * Eagle / Llama
  * Produces a rich semantic embedding of the trajectory concept

---

### **3.3 Critic Head**

A lightweight MLP:

```
Linear → LayerNorm → SiLU → Linear → Value (scalar)
```

#### **Loss**

```
L_critic = MSE(V_pred, V_target)
```

`V_target` derived from environment feedback (success_flag, -distance_to_goal).

---

### **3.4 Memory Readout (Retrieval & Fusion)**

We use a **Gated Transformer Decoder Block** to dynamically retrieve and fuse memories.

#### **Mechanism: Cross-Attention + MLP + Gating**

* **Query ($Q$)**: Current Trajectory Latent (Action Space)
* **Key ($K$)**: Semantic Latents in Memory (LLM Space)
* **Value ($V$)**: Compressed Trajectory Latents in Memory (Action Space)

$$
\text{Attn} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

$$
\text{Output} = \text{Gate}(\text{MLP}(\text{Attn} + Q))
$$

* **Zero-Init Gating**: Ensures the memory module starts with 0 influence and gradually learns to intervene.

---

### **3.5 Memory Bank**

Stores and retrieves:

* **Semantic Latent (Key)** → for intent matching
* **Trajectory Latent (Value)** → for control guidance
* **Critic Value** → for prioritization

**Storage Optimization**: We explicitly **drop** the raw (T, D) latents from storage.

Each entry:

```
{
  "semantic": (1, D_llm),     # Key
  "traj": (1, D_hidden),      # Value
  "value": scalar
}
```

---

## **4. Implementation Details**

---

### **4.1 File Structure**

```
gr00t/
│
├── model/
│   ├── action_memory/
│   │   └── memory_module.py
│   │       - ActionMemory
│   │       - TrajectoryCompressor (Transformer Encoder)
│   │       - MemoryReadoutBlock (Gated Cross-Attn)
│   │       - CriticHead
│   │
│   └── gr00t_n1.py
│       - Main VLA integration with hook injection
```

---

### **4.2 Data Flow**

```
Actions (B, T, D_action)
      ↓ ActionEncoder (existing)
Raw Latents (B, T, D)
      ↓ TrajectoryCompressor (Transformer)
Trajectory Latent (B, 1, D_hidden) ──────┐
      ↓ Projector → LLM                  │ (Query)
Semantic Latent (B, 1, D_llm)            │
      ↓ Critic                           │
Critic Value (scalar)                    │
                                         ▼
                                  Memory Retrieval
                                  (Key: Semantic, Value: Trajectory)
                                         ↓
                                  Fused Context
```

Output dict:

```python
{
    "critic_value": v,
    "semantic_latent": semantic,
    "traj_latent": traj,
    "retrieved_memory": fused_mem
}
```

---

### **4.3 Training Objective**

$$
L_{\text{total}} = L_{\text{action-diffusion}} + \lambda \cdot L_{\text{critic}}
$$

where:

* **L_action-diffusion** – standard Diffusion Policy loss
* **L_critic** – MSE on value prediction

`λ` controls influence of the critic.

---

## **5. Future Considerations**

---

### **5.1 Contrastive Alignment**

Add contrastive losses to align:

```
Semantic Latent ↔ Text Embedding (task description)
```

This improves grounding:

* “pick up cup”
* “open drawer”
* “push block to target”

---
