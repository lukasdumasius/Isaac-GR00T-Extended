# **Action Memory: Architecture & Design**

## **1. Overview**

This document describes the design of the **Action Memory Module** for the **GR00T VLA** model.

The goal is to bridge the gap between:

* **Low-level control** (continuous actions)
* **High-level semantics** (LLM reasoning)

by introducing a **structured, value-aware memory system** that operates at the **trajectory** level.

Instead of storing raw frame-by-frame actions, we:

1. **Encode entire trajectories** into compact semantic latents
2. **Store** only semantic intent and compressed control blueprints
3. **Retrieve** relevant memories dynamically using **Cross-Attention** inside the Action Head

---

## **2. Core Philosophy**

### **2.1 Trajectory as the Atomic Unit**

A single action or frame is usually not semantically meaningful. A *trajectory* represents interpretable concepts such as:

* “pick up the cup”
* “close the drawer”
* “push object to target”

Therefore, the Action Memory operates at the **trajectory** level: each memory item corresponds to a short, coherent **trajectory slice**.

---

### **2.2 Dual-Path Representation**

We maintain two parallel but coupled representations:

#### **(1) Raw Path → High-Frequency Control**

* Per-step latents: `(B, T, D)`
* Preserve smoothness, kinematic continuity, and fine-grained control for **diffusion policies**
* Used **inside the policy** for decoding, but **not stored long-term** (to save memory)

#### **(2) Semantic Path → High-Level Reasoning**

* Compress each trajectory slice into a single latent: `(B, 1, D_hidden)`
* Project into LLM space for semantic reasoning and memory indexing
* Represents **intent / behavior pattern**, not raw motor commands

The **Memory Bank** is built from the **Semantic Path** (as keys) and the compressed trajectory latents (as values).

---

## **3. Architecture Components**

---

### **3.1 Trajectory Encoder (Causal Transformer)**

Encodes raw action sequences into a single **trajectory latent**.

#### **Input**

```text
Raw Action Latents: (B, T, D_action)
```

#### **Mechanism**

* Prepend a **[CLS] token**
* Add **learnable positional embeddings**
* Pass through a **Transformer Encoder** with causal structure (can attend to past, not future)
* Use the **[CLS] output** as the trajectory summary

#### **Output**

```text
Trajectory Latent: (B, 1, D_hidden)
```

This latent is the compressed **control blueprint** for the trajectory slice.

---

### **3.2 LLM Projection (Semantic Path)**

The trajectory latent is projected into the **LLM embedding space** to obtain a semantic representation.

* **Projection Layer**: `D_hidden → D_llm`
* **LLM Backbone (Frozen)**: e.g., Eagle / LLaMA encoder tower (or similar)

  * Input: Projected trajectory latent `(B, 1, D_llm)`
  * Output: Semantically enriched latent `(B, 1, D_llm)`

This **Semantic Latent** is used as the **Key** in the Memory Bank.

---

### **3.3 Memory Readout (Retrieval & Fusion)**

Inside the **Action Head** (diffusion policy), we use **Cross-Attention** to retrieve and fuse relevant memories.

* **Query (`Q`)**: Intermediate action latents during diffusion denoising (from the Action Head)
* **Key (`K`)**: Semantic Latents stored in the Memory Bank (LLM space)
* **Value (`V`)**: Trajectory Latents stored in the Memory Bank (Action space)

#### **Cross-Attention (Training: Causal Mask; Inference: No Mask)**

Training uses a causal mask over the **memory time dimension** to prevent future leakage when we simulate incremental memory growth within an episode.

Let:

* `Q`: query from current step
* `K`: concatenated semantic keys from all past slices
* `V`: corresponding trajectory values

Then:

$$
\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top + M}{\sqrt{d}}\right)V
$$

where the **causal mask** $M$ over memory items is:

$$
M_{ij} =
\begin{cases}
0 & \text{if } i \ge j \quad (\text{can attend to past}) \\
-\infty & \text{if } i < j \quad (\text{cannot attend to future})
\end{cases}
$$

* **Training**: `causal_mask=True` (simulate incremental KV cache)
* **Inference**: `causal_mask=False` (we are already step-by-step; no future exists)

#### **Gated Fusion (Planned)**

The **current implementation** uses standard cross-attention.
The **planned upgrade** is a **Gated Cross-Attention Block** that:

1. Applies cross-attention to compute a memory-informed latent
2. Passes it through an MLP
3. Uses a **zero-initialized gate** to control how much the memory influences the Action Head

Planned form:

```python
Output = Gate(MLP(Attn(Q, K, V) + Q))
# Gate is zero-initialized, learned during training
```

This allows the model to **start with no memory influence** and gradually learn **when** to trust memory.

---

### **3.4 Memory Bank (KV Cache)**

The **Memory Bank** is an incremental **KV cache** analogous to an LLM’s KV cache, but at **trajectory-slice granularity**.

* **Key**: Semantic Latent (LLM space)
* **Value**: Trajectory Latent (action space)

#### **3.4.1 Training: Per-Episode Cache**

* Cache is **cleared** at the start of each episode
* Within an episode, it **grows incrementally** as we process trajectory slices

Example:

```text
Episode 1 start:
  KV_cache = []

Forward 1 (slice 1):
  → KV1 = (Semantic1, Traj1)
  KV_cache = [KV1]

Forward 2 (slice 2):
  → KV2 = (Semantic2, Traj2)
  KV_cache = [KV1, KV2]

...
Episode 1 end → cache cleared
Episode 2 start → KV_cache = []
```

This keeps training stable and memory bounded by **episode length**.

---

#### **3.4.2 Inference: Continuous Cache with Eviction**

During **deployment**, we do **not** have clear episode boundaries.
Instead, we use a **fixed-capacity KV cache** with **mandatory eviction**:

* `max_capacity = N_max` (e.g., 100 trajectory slices)
* Each new trajectory slice produces a KV pair
* If cache is full when a new KV arrives → **must evict one old KV**

Eviction strategies (selected by design):

1. **FIFO (First-In-First-Out)** – simplest
2. **Attention-based** – evict KV with lowest average attention score
3. **Value / heuristic-based** – evict KV that appears least useful (e.g., old or low-relevance tasks)

Example (FIFO):

```text
Start: KV_cache = [], MAX = 10

Forward 1..10:
  Fill cache to [KV1, KV2, ..., KV10]

Forward 11:
  Cache full → evict KV1
  KV_cache = [KV2, KV3, ..., KV10, KV11]

Forward 12:
  Evict KV2
  KV_cache = [KV3, KV4, ..., KV11, KV12]

...
Run indefinitely with rolling window of last 10 slices
```

This enables **lifelong operation** without ever clearing memory.

---

## **4. Implementation Details**

---

### **4.1 File Structure**

```text
gr00t/
│
├── model/
│   ├── action_memory/
│   │   └── memory_module.py
│   │       - ActionMemory
│   │       - TrajectoryCompressor (Transformer Encoder)
│   │       - MemoryReadoutBlock (Cross-Attention / Gated Cross-Attention)
│   │
│   └── gr00t_n1.py
│       - Main VLA integration (vision + language + action_head + memory)
```

---

### **4.2 Data Flow: Training (Per-Batch)**

We assume each batch contains **B episodes (rollouts)**.

#### **Step 1: Build Per-Sample KV Caches**

```python
for batch in vla_dataloader:
    vision, text, actions, ground_truth_actions = batch  # B samples
    batch_size = vision.shape[0]
    
    batch_kv_caches = []
    
    for sample_idx in range(batch_size):
        # Full trajectory for this sample
        sample_actions = actions[sample_idx]  # (T_total, D_action)
        
        # 1) Slice into windows (trajectory units)
        trajectory_slices = slice_trajectory(sample_actions, window_size=10)
        
        # 2) Initialize KV cache for this sample
        sample_kv_cache = {"semantic_keys": [], "traj_values": []}
        
        # 3) Encode each slice → KV pair
        for traj_slice in trajectory_slices:
            traj_latent = trajectory_compressor(traj_slice.unsqueeze(0))  # (1, 1, D_hidden)
            semantic_latent = llm_backbone(traj_latent)                   # (1, 1, D_llm)
            
            sample_kv_cache["semantic_keys"].append(semantic_latent)
            sample_kv_cache["traj_values"].append(traj_latent)
        
        batch_kv_caches.append(sample_kv_cache)
```

#### **Step 2: VLA Forward + Memory Retrieval**

```python
    # Forward pass through VLA backbone (vision + text + raw actions)
    outputs = vla_model(vision, text, actions)
    intermediate_latents = outputs["intermediate_latents"]  # (B, 1, D_model)
    
    retrieved_memories = []
    
    for sample_idx in range(batch_size):
        sample_kv = batch_kv_caches[sample_idx]
        
        if len(sample_kv["semantic_keys"]) > 0:
            mem_keys = torch.cat(sample_kv["semantic_keys"], dim=1)   # (1, N, D_llm)
            mem_values = torch.cat(sample_kv["traj_values"], dim=1)   # (1, N, D_hidden)
            
            query = intermediate_latents[sample_idx:sample_idx+1]     # (1, 1, D_model)
            
            # Cross-attention with CAUSAL MASK (within-sample memory time)
            retrieved = cross_attention(
                query=query,
                keys=mem_keys,
                values=mem_values,
                causal_mask=True
            )  # (1, 1, D_hidden)
        else:
            retrieved = torch.zeros_like(intermediate_latents[sample_idx:sample_idx+1])
        
        retrieved_memories.append(retrieved)
    
    retrieved_memories = torch.cat(retrieved_memories, dim=0)  # (B, 1, D_hidden)
```

#### **Step 3: Fuse Memory into Action Head and Compute Loss**

```python
    # Pass retrieved memory into action head (e.g., concatenation or FiLM-style conditioning)
    final_outputs = vla_model.action_head(
        intermediate_latents,
        retrieved_memories
    )
    
    # Standard diffusion loss
    loss = compute_diffusion_loss(
        final_outputs["predicted_noise"],
        ground_truth_noise
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # KV caches are discarded at the end of the batch (training episodes)
```

---

### **4.3 Training Objective (No Critic)**

The core training objective is **standard diffusion loss**, with memory only influencing the **conditioning**:

$$
L_{\text{total}} = L_{\text{action-diffusion}}
$$

$$
L_{\text{action-diffusion}} = \mathbb{E}_{t, \epsilon} \left[ \lVert \epsilon - \epsilon_\theta(x_t, t, c) \rVert^2 \right]
$$

* $x_t$: noisy action at timestep $t$
* $\epsilon$: ground-truth noise
* $\epsilon_\theta$: predicted noise
* $c$: conditioning (vision + language + **retrieved memory**)

**Key Point**: The only change vs. baseline is that the conditioning incorporates **history-aware memory context** derived from past trajectories.

---

### **4.4 Inference: Continuous Operation with Fixed-Capacity Cache**

```python
def inference_continuous(vision_stream, text, max_cache_size=100):
    """
    Continuous inference with a fixed-capacity KV cache.
    """
    vla_model.eval()
    
    kv_cache = {"semantic_keys": [], "traj_values": []}
    generated_actions = []
    
    for t in range(len(vision_stream)):
        current_vision = vision_stream[t]  # (C, H, W)
        
        # Build trajectory slice from recent generated actions
        if len(generated_actions) >= 10:
            recent_slice = torch.stack(generated_actions[-10:])  # (10, D_action)
        elif len(generated_actions) > 0:
            recent_slice = torch.stack(generated_actions)        # (<10, D_action)
        else:
            recent_slice = None
        
        # 1) If we have history, update KV cache
        if recent_slice is not None:
            with torch.no_grad():
                traj_latent = trajectory_compressor(recent_slice.unsqueeze(0))  # (1, 1, D_hidden)
                semantic_latent = llm_backbone(traj_latent)                     # (1, 1, D_llm)
                
                # Evict if full
                if len(kv_cache["semantic_keys"]) >= max_cache_size:
                    # Simple FIFO eviction
                    kv_cache["semantic_keys"].pop(0)
                    kv_cache["traj_values"].pop(0)
                
                # Append new KV
                kv_cache["semantic_keys"].append(semantic_latent)
                kv_cache["traj_values"].append(traj_latent)
        
        # 2) Forward pass through VLA backbone
        with torch.no_grad():
            outputs = vla_model(current_vision.unsqueeze(0), text, None)
            intermediate_latent = outputs["intermediate_latents"]  # (1, 1, D_model)
            
            # 3) Retrieve from cache (no causal mask needed in online inference)
            if len(kv_cache["semantic_keys"]) > 0:
                mem_keys = torch.cat(kv_cache["semantic_keys"], dim=1)   # (1, N, D_llm)
                mem_values = torch.cat(kv_cache["traj_values"], dim=1)   # (1, N, D_hidden)
                
                retrieved = cross_attention(
                    query=intermediate_latent,
                    keys=mem_keys,
                    values=mem_values,
                    causal_mask=False
                )
            else:
                retrieved = torch.zeros_like(intermediate_latent)
            
            # 4) Action head with memory fusion
            action = vla_model.action_head(intermediate_latent, retrieved)  # (1, 1, D_action)
        
        generated_actions.append(action.squeeze(0))
    
    return torch.stack(generated_actions)  # (T, D_action)
```

---

## **5. Future Directions (Memory Core)**

### **5.1 Gated Cross-Attention**

Planned upgrade for `MemoryReadoutBlock`:

```python
class MemoryReadoutBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Zero-init gate
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    
    def forward(self, query, memory_keys, memory_values):
        # query: (B, 1, D), memory_*: (B, N, D)
        attn_out, _ = self.cross_attn(
            query, memory_keys, memory_values
        )  # (B, 1, D)
        
        x = self.norm1(query + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        gate_value = torch.sigmoid(self.gate(x))  # (B, 1, 1)
        return gate_value * x
```

Benefits:

* Starts with **no memory influence** (gate ≈ 0)
* Learns **when** memory is useful
* Reduces risk of destabilizing early training

---

## **6. Optional Critic Extension (Mentioned Only Here)**

> **Note:** The core Action Memory design above works **without** any Critic.
> The Critic is an **optional extension**, not part of the main module.

A possible extension is to add a **lightweight Critic MLP** on top of the **Semantic Latents** to provide **advantage-weighted training**:

* Input: Aggregated semantic latent for a trajectory (e.g., mean over slices)
* Output: Scalar value estimate $V_{\text{pred}}$ for trajectory quality

Two-stage training (optional):

1. **Stage 1 – Train Critic Only**

   * Loss:
     $$
     L_{\text{critic}} = \lVert V_{\text{pred}} - V_{\text{target}} \rVert^2
     $$
   * Trains a small MLP head on semantic latents (LLM backbone kept frozen)

2. **Stage 2 – Train VLA with Advantage Weighting**

   * Critic is frozen
   * Compute advantage:
     $$
     A = V_{\text{target}} - V_{\text{pred}}
     $$
   * Weight diffusion loss per sample:
     $$
     L_{\text{total}} = \mathbb{E}[A \cdot L_{\text{action-diffusion}}]
     $$

This may help emphasize trajectories where the model underperforms, but:

* Requires a reliable reward / success signal
* Increases complexity
* Is **not required** for the core Action Memory to work
