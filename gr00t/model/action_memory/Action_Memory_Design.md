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
3. **Store** them in a **Memory Bank** for long-horizon planning and retrieval

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

We preserve **two parallel pathways**:

#### **Raw Path → High-Frequency Control**

* Retains (B, T, D) per-step latents
* Ensures smoothness and kinematic fidelity for Diffusion Policies
* Used for reconstruction and replay

#### **Semantic Path → High-Level Reasoning**

* Compress trajectory → single latent (B, 1, D)
* Used for LLM reasoning + memory indexing
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

#### **Input**

* LLM-processed semantic latent

#### **Output**

* Scalar value

  ```
  v ∈ ℝ  (success probability / stability score)
  ```

#### **Loss**

```
L_critic = MSE(V_pred, V_target)
```

`V_target` derived from environment feedback (success_flag, -distance_to_goal).

---

### **3.4 Memory Bank**

Stores and retrieves:

* **Semantic Latent** → for reasoning & retrieval
* **Raw Trajectory Latents** → for replay
* **Critic Value** → for prioritization

Each entry:

```
{
  "semantic": (1, D),
  "raw": (T, D_action),
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
Trajectory Latent (B, 1, D_hidden)
      ↓ Projector → LLM
Semantic Latent (B, 1, D_llm)
      ↓ Critic
Critic Value (scalar)
```

Output dict:

```python
{
    "critic_value": v,
    "semantic_latent": semantic,
    "raw_latents": raw_latents,
}
```

---

### **4.3 Training Objective**

$$
L_{\text{total}} = L_{\text{action\_diffusion}} + \lambda \cdot L_{\text{critic}}
$$

where:

* **L_action_diffusion** – standard Diffusion Policy loss
* **L_critic** – MSE on value prediction

`λ` controls influence of the critic.

---

## **5. Future Considerations**

---

### **5.1 Retrieval Mechanism**

Implement vector search for semantic latents:

* Cosine similarity
* FAISS / ScaNN / local attention retrieval
* Task-conditioned filtering using LLM queries

---

### **5.2 Contrastive Alignment**

Add contrastive losses to align:

```
Semantic Latent ↔ Text Embedding (task description)
```

This improves grounding:

* “pick up cup”
* “open drawer”
* “push block to target”

---

### **5.3 Multi-Trajectory Value Aggregation**

Consider storing **prefixes**, **suffixes**, and **failure trajectories**, enabling:

* recovery heuristics
* alternative plans
* counterfactual reasoning
