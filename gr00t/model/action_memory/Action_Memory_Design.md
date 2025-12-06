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

* "pick up the cup"
* "close the drawer"
* "push object to target"

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
* Represents "intent," not motor signals

---

### **2.3 Critic-Guided Optimization**

A lightweight **Critic** evaluates trajectory quality (success probability, stability, etc.).

The Critic's gradients:

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

A **lightweight MLP** that operates on the Semantic Latent (which has already been processed by the LLM backbone).

#### **Architecture**

```
Semantic Latent (B, 1, D_llm) [from LLM backbone]
      ↓
Linear(D_llm → D_hidden)
      ↓
LayerNorm
      ↓
SiLU
      ↓
Linear(D_hidden → D_hidden)
      ↓
SiLU
      ↓
Linear(D_hidden → 1)
      ↓
Critic Value (B, 1) [scalar per trajectory]
```

**Key Insight**: The Critic does NOT need to be a separate VLM instance. The Semantic Latent already contains rich, high-level information about Vision + Text + Actions processed by the LLM. The Critic simply learns a lightweight mapping from this semantic representation to a quality score.

#### **Input**

* Semantic Latent (already processed by LLM Backbone A)

#### **Output**

* Critic Value (scalar) - predicted success/quality score

#### **Loss**

```
L_critic = MSE(V_pred, V_target)
```

`V_target` derived from environment feedback (success_flag, -distance_to_goal).

---

### **3.4 Memory Readout (Retrieval & Fusion)**

We use a **Gated Transformer Decoder Block** to dynamically retrieve and fuse memories **within the Action Head**.

**⚠️ Status**: Gated Cross-Attention mechanism is **TO BE IMPLEMENTED** (target: NeurIPS 2025).

Current simplified version uses standard cross-attention. Full gated mechanism will be added.

**⚠️ Important - Causal Masking**: 
- **Training**: Use causal mask to prevent attention to future KV pairs
- **Inference**: No mask needed (generating autoregressively, no future exists)

#### **Mechanism: Cross-Attention + MLP + Gating**

* **Query ($Q$)**: Action Head Intermediate Latents (during diffusion denoising steps)
* **Key ($K$)**: Semantic Latents stored in Memory Bank (LLM Space)
* **Value ($V$)**: Trajectory Latents stored in Memory Bank (Action Space)

**Cross-Attention (with Causal Mask in Training)**:
$$
\text{Attn} = \text{Softmax}\left(\frac{QK^T + M}{\sqrt{d}}\right)V
$$

where $M$ is the causal mask:
$$
M_{ij} = \begin{cases}
0 & \text{if } i \geq j \text{ (can attend to past)} \\
-\infty & \text{if } i < j \text{ (cannot attend to future)}
\end{cases}
$$

**Gated Fusion (TO BE IMPLEMENTED)**:
$$
\text{Output} = \text{Gate}(\text{MLP}(\text{Attn} + Q))
$$

* **Zero-Init Gating**: Ensures the memory module starts with 0 influence and gradually learns to intervene.

**Key Insight**: The Action Head's intermediate latents (noisy states during diffusion) query the memory bank to retrieve relevant past trajectory patterns, which are then fused back into the denoising process via cross-attention.

---

### **3.5 Memory Bank (KV Cache)**

The Memory Bank works as an **incremental KV cache**, similar to LLM's KV cache mechanism.

**Key Properties**:

1. **Incremental Growth**: KV pairs accumulate over time within an episode.
   - Each forward pass generates **new** KV pairs from the latest trajectory slice
   - These are **appended** to existing KV pairs from previous steps
   - Similar to how LLM KV cache grows as you process more tokens

2. **Per-Episode Cache**: Each episode maintains its own growing KV cache.
   - Step 1: Generate KV₁ from slice₁ → Cache = [KV₁]
   - Step 2: Generate KV₂ from slice₂ → Cache = [KV₁, KV₂]
   - Step 3: Generate KV₃ from slice₃ → Cache = [KV₁, KV₂, KV₃]
   - ...

3. **Query Growing Cache**: The Action Head's intermediate latent queries the **entire accumulated cache**.
   - At step t, query attends to all KV pairs from steps [1, 2, ..., t]
   - More context available as episode progresses

4. **Cache Management**:
   - **Training**: Cache is **cleared** at the start of each new episode
     - Episode boundaries are well-defined in training data
     - Clean separation between different task instances
     - Cache size is bounded by episode length (usually < 100 steps)
   
   - **Inference**: Cache has **fixed capacity** with **mandatory eviction**
     - Set maximum capacity (e.g., 100 KV pairs)
     - When cache is full and new KV arrives: **must evict** one old KV
     - No clearing - cache persists across tasks indefinitely
     - Eviction strategies:
       * **FIFO (First-In-First-Out)**: Remove oldest KV pair
       * **Attention-based**: Remove KV with lowest attention score
       * **Value-based**: Remove KV with lowest Critic value
     - Allows continuous operation without episode boundaries

**Analogy to LLM KV Cache**: 
- LLM: Process token → Generate (K, V) → Append to cache → Attend over all cached KVs
- Action Memory: Process trajectory slice → Generate (Semantic, Trajectory) → Append to cache → Attend over all cached KVs

Example for one episode (Training):

```
Episode 1 Start: KV_cache = []

Forward 1: Generate (Semantic₁, Traj₁)
  KV_cache = [(Semantic₁, Traj₁)]

Forward 2: Generate (Semantic₂, Traj₂)
  KV_cache = [(Semantic₁, Traj₁), (Semantic₂, Traj₂)]

...

Episode 1 End

Episode 2 Start: KV_cache = []  # Cleared for training

Forward 1: Generate (Semantic₁, Traj₁)
  KV_cache = [(Semantic₁, Traj₁)]

...
```

Example for continuous operation (Inference):

```
Inference Start: KV_cache = [], max_capacity = 10

Forward 1-10: KV_cache grows
  Forward 1: KV_cache = [KV₁]
  Forward 2: KV_cache = [KV₁, KV₂]
  ...
  Forward 10: KV_cache = [KV₁, KV₂, ..., KV₁₀]  # Full!

Forward 11: Cache FULL, must evict
  New KV₁₁ arrives → Evict KV₁ (oldest)
  KV_cache = [KV₂, KV₃, ..., KV₁₀, KV₁₁]  # Still size 10

Forward 12: Cache FULL, must evict
  New KV₁₂ arrives → Evict KV₂ (oldest)
  KV_cache = [KV₃, KV₄, ..., KV₁₁, KV₁₂]  # Still size 10

Continue indefinitely: Always evict when full, never clear
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

#### **Memory Bank Population (Training)**

KV cache **accumulates incrementally** across forward passes within an episode.

```
Episode with multiple forward passes:

Forward Pass 1 (latest trajectory slice):
  New Trajectory Slice (T_slice steps)
      ↓ ActionEncoder
  Raw Latents (1, T_slice, D)
      ↓ TrajectoryCompressor
  Trajectory Latent (1, 1, D_hidden) ───────┐
      ↓ Projector → LLM                     │
  Semantic Latent (1, 1, D_llm) ────┐       │
      ↓                             │       │
      ↓ Lightweight Critic MLP      │       │
  Critic Value (scalar)             │       │
                                    │       │
  Generate NEW KV Pair:             │       │
    - Key: Semantic Latent ◄────────┘       │
    - Value: Trajectory Latent ◄────────────┘
  
  KV_cache = [KV₁]  # First entry

---

Forward Pass 2 (next trajectory slice):
  Same process → Generate KV₂
  
  KV_cache = [KV₁, KV₂]  # Appended to cache

---

Forward Pass t (current trajectory slice):
  Same process → Generate KVₜ
  
  KV_cache = [KV₁, KV₂, ..., KVₜ]  # Growing cache

Batch Processing:
For batch size B, each sample maintains its own growing KV cache.
```

#### **Memory Retrieval & Fusion (Action Generation)**

Query attends to **all accumulated KV pairs** in the cache **with causal masking**.

**⚠️ Important**: During training, use **causal mask** to prevent looking into future KV pairs.

```
At Forward Pass t:

KV Cache: [(K₁, V₁), (K₂, V₂), ..., (Kₜ, Vₜ)]
          (accumulated from all previous steps)
                          ↓
Vision + Text → Backbone → Action Head (Diffusion)
                                ↓
                    Intermediate Latent at step t (Query)
                                ↓
                                ▼
                    Cross-Attention with Causal Mask
                    Q_t = Intermediate Latent (current step t)
                    K = [K₁, K₂, ..., Kₜ] (all Semantic Latents)
                    V = [V₁, V₂, ..., Vₜ] (all Trajectory Latents)
                    
                    Mask: Q_t can only attend to K₁...Kₜ (not Kₜ₊₁, Kₜ₊₂, ...)
                          Prevents information leakage from future
                                ↓
                        Retrieved Memory
                        (weighted sum over past trajectories only)
                                ↓
                    Fused into Action Generation
                                ↓
                    Predicted Actions
                                ↓
                    L_action_base = MSE(Predicted, GT)

Next forward: KV cache grows, more context for retrieval

---

Stage 1 Training (Critic Only):
L_critic = MSE(V_pred, V_target)  [Critic on Semantic Latents]
(Trains Critic independently)

Stage 2 Training (VLA with Frozen Critic):
Advantage (A) = V_target - V_pred  [from frozen Critic]
L_total = mean_over_batch(A · L_action_base)
(Trains VLA with advantage-weighted action loss)
```

---

### **4.3 Training Objective**

The training is divided into **two stages**:

---

#### **Stage 1: Pre-train the Critic (Independent)**

In the first stage, we train **only the Critic** to predict trajectory quality:

$$
L_{\text{critic}} = \text{MSE}(V_{\text{pred}}, V_{\text{target}}) = \| V_{\text{pred}} - V_{\text{target}} \|^2
$$

where:
* $V_{\text{pred}}$ is the predicted value from the Independent VLM Critic
* $V_{\text{target}}$ is the ground-truth reward/success signal from the dataset

**Purpose**: Train the Critic to be an accurate "judge" of trajectory quality **before** using it to guide the VLA.

**Implementation (Stage 1):**

```python
# Stage 1: Train Critic Only
# Each sample maintains its own memory bank

for batch in critic_dataloader:
    # batch contains B rollout/episodes
    vision, text, actions, target_value = batch  # Each shape: (B, ...)
    batch_size = vision.shape[0]
    
    # Per-sample memory banks (list of lists)
    batch_memory_banks = []
    
    for sample_idx in range(batch_size):
        # Get this sample's full trajectory
        sample_actions = actions[sample_idx]  # (T_total, D_action)
        
        # Slice trajectory into windows
        trajectory_slices = slice_trajectory(sample_actions, window_size=10)
        # Returns list of slices: [(T_slice, D_action), ...]
        
        sample_memory = {"semantic_keys": [], "traj_values": []}
        
        # Encode each slice
        for traj_slice in trajectory_slices:
            with torch.no_grad():
                traj_latent = trajectory_compressor(traj_slice.unsqueeze(0))  # (1, 1, D_hidden)
                semantic_latent = llm_backbone(traj_latent)  # (1, 1, D_llm)
                
                # Add to THIS sample's memory
                sample_memory["semantic_keys"].append(semantic_latent)
                sample_memory["traj_values"].append(traj_latent)
        
        batch_memory_banks.append(sample_memory)
    
    # Train Critic on the full trajectories
    pred_value = critic_model(
        vision, text, actions
    )  # (B,) - one value per rollout
    
    # Compute critic loss
    loss_critic = F.mse_loss(pred_value, target_value)
    
    # Backpropagation (only updates Critic weights)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
```

---

#### **Stage 2: Train VLA with Advantage-Weighted Action Loss**

In the second stage, we **freeze the Critic** and use it to compute advantages that weight the VLA's action loss:

$$
L_{\text{total}} = A \cdot L_{\text{action-diffusion}}
$$

where:

**Action Diffusion Loss (Base):**

$$
L_{\text{action-diffusion}} = \mathbb{E}_{t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, c) \|^2 \right]
$$

* $x_t$ is the noisy action at timestep $t$
* $\epsilon$ is the ground-truth noise
* $\epsilon_\theta$ is the predicted noise from the diffusion model
* $c$ is the conditioning (vision + language + retrieved memory)

**Advantage Weighting:**

$$
A = V_{\text{target}} - V_{\text{pred}}
$$

or alternatively (normalized):

$$
A = \frac{V_{\text{target}} - V_{\text{pred}}}{\text{std}(V_{\text{target}})}
$$

**Purpose**:
* **High-quality trajectories** ($V_{\text{target}}$ high, $V_{\text{pred}}$ accurate) → **Lower advantage** → Less gradient emphasis (already learned well)
* **Low-quality trajectories** ($V_{\text{target}}$ low) → Can be down-weighted to avoid learning bad examples
* **Prediction errors** (large $|A|$) → **Higher gradient** → Forces model to learn from mistakes

**Implementation (Stage 2):**

```python
# Stage 2: Train VLA with Advantage Weighting
# Critic is frozen, Memory Bank is fully populated
critic_model.eval()

for batch in vla_dataloader:
    vision, text, actions, ground_truth_actions, target_value = batch  # B samples
    
    # Forward pass through VLA
    outputs = vla_model(vision, text, actions)
    
    # Retrieve from GLOBAL Memory Bank
    # Each sample in batch queries the same memory bank
    intermediate_latents = outputs["intermediate_latents"]  # (B, 1, D)
    
    # Cross-attention: (B, 1, D) queries (N, D_llm) keys -> (B, N) attention
    # Then weighted sum over (N, D_hidden) values -> (B, 1, D_hidden)
    all_semantic_keys = torch.cat(global_memory_bank["semantic_keys"])  # (N, D_llm)
    all_traj_values = torch.cat(global_memory_bank["traj_values"])      # (N, D_hidden)
    
    retrieved_memory = cross_attention(
        query=intermediate_latents,
        keys=all_semantic_keys,
        values=all_traj_values
    )  # (B, 1, D_hidden) - different for each sample
    
    # Continue diffusion with fused memory
    final_outputs = vla_model.action_head(retrieved_memory)
    
    # Compute base action diffusion loss
    loss_action_base = compute_diffusion_loss(
        final_outputs["predicted_noise"], 
        ground_truth_noise
    )  # (B,)
    
    # Get advantage from frozen Critic (computed per sample)
    with torch.no_grad():
        pred_value = critic_model(vision, text, actions)  # (B,)
        advantage = target_value - pred_value  # (B,)
        # Optional: normalize
        advantage = advantage / (target_value.std() + 1e-8)
    
    # Weight action loss by advantage (element-wise)
    loss_action_weighted = advantage * loss_action_base  # (B,)
    
    # Total loss (mean over batch)
    loss_total = loss_action_weighted.mean()
    
    # Backpropagation (only updates VLA weights)
    vla_optimizer.zero_grad()
    loss_total.backward()
    vla_optimizer.step()
```

---

#### **Summary: Two-Stage Training**

| Stage | Model Trained | Loss Function | Critic Role | Memory Bank |
|-------|--------------|---------------|-------------|-------------|
| **Stage 1** | Critic VLM | $L_{\text{critic}} = \text{MSE}(V_{\text{pred}}, V_{\text{target}})$ | Being trained | Built per-sample |
| **Stage 2** | VLA (Action Head) | $L_{\text{total}} = A \cdot L_{\text{action}}$ | Frozen, provides Advantage | Used for retrieval |

---

### **4.4 Training Pseudocode**

---

#### **Basic Version (Currently Implemented)**

The basic training version trains only the VLA with memory retrieval, **without Critic supervision**.

```python
# Basic Training: VLA with Memory Retrieval (No Critic)

for batch in vla_dataloader:
    # batch contains B rollout/episodes
    vision, text, actions, ground_truth_actions = batch
    batch_size = vision.shape[0]
    
    # Build KV cache for each sample (dynamically during forward)
    batch_kv_caches = []
    
    for sample_idx in range(batch_size):
        sample_actions = actions[sample_idx]  # (T_total, D_action)
        
        # Slice trajectory into windows
        trajectory_slices = slice_trajectory(sample_actions, window_size=10)
        
        sample_kv_cache = {"semantic_keys": [], "traj_values": []}
        
        # Generate KV pairs for this sample
        for traj_slice in trajectory_slices:
            traj_latent = trajectory_compressor(traj_slice.unsqueeze(0))
            semantic_latent = llm_backbone(traj_latent)
            
            # Append to this sample's KV cache
            sample_kv_cache["semantic_keys"].append(semantic_latent)
            sample_kv_cache["traj_values"].append(traj_latent)
        
        batch_kv_caches.append(sample_kv_cache)
    
    # Forward pass through VLA
    outputs = vla_model(vision, text, actions)
    intermediate_latents = outputs["intermediate_latents"]  # (B, 1, D)
    
    # Retrieve from per-sample KV caches
    retrieved_memories = []
    for sample_idx in range(batch_size):
        sample_kv = batch_kv_caches[sample_idx]
        
        if len(sample_kv["semantic_keys"]) > 0:
            mem_keys = torch.cat(sample_kv["semantic_keys"])  # (N, D_llm)
            mem_values = torch.cat(sample_kv["traj_values"])  # (N, D_hidden)
            
            query = intermediate_latents[sample_idx:sample_idx+1]  # (1, 1, D)
            
            # Standard cross-attention with CAUSAL MASK
            # Important: During training, mask future KV pairs
            # If processing step t, can only attend to KV₁...KVₜ
            retrieved = cross_attention(
                query=query,
                keys=mem_keys,
                values=mem_values,
                causal_mask=True  # Prevent information leakage
            )
        else:
            retrieved = torch.zeros_like(intermediate_latents[sample_idx:sample_idx+1])
        
        retrieved_memories.append(retrieved)
    
    retrieved_memories = torch.cat(retrieved_memories)
    
    # Generate actions with memory fusion
    final_outputs = vla_model.action_head(retrieved_memories)
    
    # Compute action diffusion loss (standard, no advantage weighting)
    loss = compute_diffusion_loss(
        final_outputs["predicted_noise"],
        ground_truth_noise
    )
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # KV cache cleared after batch (episode boundaries)
```

---

#### **Advanced Version with Critic (TODO)**

The full version includes two-stage training with Critic supervision (see Section 5.2 for implementation requirements).

**Stage 1: Train Critic**
```python
# Train lightweight Critic MLP on semantic latents
for batch in critic_dataloader:
    # Generate KV cache and aggregate semantic latents
    semantic_latents = generate_and_aggregate_semantics(batch)
    
    # Critic prediction
    pred_value = critic_head(semantic_latents)
    loss_critic = F.mse_loss(pred_value, target_value)
    
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
```

**Stage 2: Train VLA with Frozen Critic**
```python
critic_head.eval()  # Freeze

for batch in vla_dataloader:
    # Same as basic version: generate KV, retrieve, generate actions
    retrieved_memories = retrieve_from_kv_cache(batch)
    final_outputs = vla_model.action_head(retrieved_memories)
    loss_action_base = compute_diffusion_loss(...)
    
    # Compute advantage from frozen Critic
    with torch.no_grad():
        semantic_latents = aggregate_semantics(batch)
        pred_value = critic_head(semantic_latents)
        advantage = target_value - pred_value
    
    # Advantage-weighted loss
    loss_total = (advantage * loss_action_base).mean()
    
    vla_optimizer.zero_grad()
    loss_total.backward()
    vla_optimizer.step()
```

---

```python
# Stage 1: Train Critic (lightweight MLP)
# KV pairs are generated dynamically during each forward pass

for batch in critic_dataloader:
    # batch contains B rollout/episodes
    vision, text, actions, target_value = batch  # Each shape: (B, ...)
    batch_size = vision.shape[0]
    
    # Lists to collect semantic latents for Critic
    all_semantic_latents = []
    
    for sample_idx in range(batch_size):
        # Get this sample's full trajectory
        sample_actions = actions[sample_idx]  # (T_total, D_action)
        
        # Slice trajectory into windows
        trajectory_slices = slice_trajectory(sample_actions, window_size=10)
        # Returns list of slices: [(T_slice, D_action), ...]
        
        sample_semantics = []
        
        # FORWARD PASS: Generate KV pairs for this sample
        for traj_slice in trajectory_slices:
            # Encode slice → Generate one KV pair
            traj_latent = trajectory_compressor(traj_slice.unsqueeze(0))  # (1, 1, D_hidden) [Value]
            semantic_latent = llm_backbone(traj_latent)  # (1, 1, D_llm) [Key]
            
            # KV pair exists only during this forward pass
            sample_semantics.append(semantic_latent)
        
        # Aggregate semantic latents for Critic (e.g., mean or last)
        rollout_semantic = torch.cat(sample_semantics).mean(dim=0, keepdim=True)  # (1, 1, D_llm)
        all_semantic_latents.append(rollout_semantic)
    
    # Stack all rollout semantics
    semantic_latents = torch.cat(all_semantic_latents)  # (B, 1, D_llm)
    
    # Train Critic: lightweight MLP on semantic latents
    pred_value = critic_head(semantic_latents).squeeze(-1)  # (B,)
    
    # Compute critic loss
    loss_critic = F.mse_loss(pred_value, target_value)
    
    # Backpropagation (only updates Critic head weights)
    critic_optimizer.zero_grad()
    loss_critic.backward()
    critic_optimizer.step()
    
    # KV pairs are discarded after this forward pass
```

#### **Stage 2: Train VLA with Frozen Critic & Memory Retrieval**

```python
# Stage 2: Train VLA with advantage-weighting
# Critic (lightweight MLP) is frozen, memory banks are used for retrieval

critic_head.eval()

for batch in vla_dataloader:
    # batch contains B rollout/episodes
    vision, text, actions, ground_truth_actions, target_value = batch
    batch_size = vision.shape[0]
    
    # Build memory banks for this batch (same as Stage 1)
    batch_memory_banks = []
    all_semantic_latents = []
    
    for sample_idx in range(batch_size):
        sample_actions = actions[sample_idx]
        trajectory_slices = slice_trajectory(sample_actions, window_size=10)
        
        sample_memory = {"semantic_keys": [], "traj_values": []}
        sample_semantics = []
        
        for traj_slice in trajectory_slices:
            with torch.no_grad():
                traj_latent = trajectory_compressor(traj_slice.unsqueeze(0))
                semantic_latent = llm_backbone(traj_latent)
                sample_memory["semantic_keys"].append(semantic_latent)
                sample_memory["traj_values"].append(traj_latent)
                sample_semantics.append(semantic_latent)
        
        batch_memory_banks.append(sample_memory)
        
        # Average semantic for Critic
        rollout_semantic = torch.cat(sample_semantics).mean(dim=0, keepdim=True)
        all_semantic_latents.append(rollout_semantic)
    
    semantic_latents = torch.cat(all_semantic_latents)  # (B, 1, D_llm)
    
    # Forward pass through VLA
    outputs = vla_model(vision, text, actions)
    intermediate_latents = outputs["intermediate_latents"]  # (B, 1, D)
    
    # Retrieve from per-sample memory banks
    retrieved_memories = []
    for sample_idx in range(batch_size):
        # Get this sample's memory
        sample_memory = batch_memory_banks[sample_idx]
        sample_keys = torch.cat(sample_memory["semantic_keys"])  # (N_i, D_llm)
        sample_values = torch.cat(sample_memory["traj_values"])  # (N_i, D_hidden)
        
        # Query with this sample's intermediate latent
        query = intermediate_latents[sample_idx:sample_idx+1]  # (1, 1, D)
        
        # Cross-attention retrieval
        retrieved = cross_attention(
            query=query,
            keys=sample_keys,
            values=sample_values
        )  # (1, 1, D_hidden)
        retrieved_memories.append(retrieved)
    
    retrieved_memories = torch.cat(retrieved_memories)  # (B, 1, D_hidden)
    
    # Continue diffusion with fused memory
    final_outputs = vla_model.action_head(retrieved_memories)
    
    # Compute base action loss
    loss_action_base = compute_diffusion_loss(
        final_outputs["predicted_noise"],
        ground_truth_noise
    )  # (B,)
    
    # Get advantage from frozen Critic (lightweight MLP)
    with torch.no_grad():
        pred_value = critic_head(semantic_latents).squeeze(-1)  # (B,)
        advantage = target_value - pred_value  # (B,)
        advantage = advantage / (target_value.std() + 1e-8)  # normalize
    
    # Weight action loss by advantage
    loss_action_weighted = advantage * loss_action_base  # (B,)
    loss_total = loss_action_weighted.mean()
    
    # Backpropagation (only updates VLA weights)
    vla_optimizer.zero_grad()
    loss_total.backward()
    vla_optimizer.step()
```

---

### **4.5 Inference Pseudocode**

During inference, the model generates actions step-by-step with **continuous KV cache** (no clearing between episodes).

```python
# Inference: Generate actions with persistent KV cache and selective eviction

def inference_continuous(vision_stream, text, max_cache_size=100):
    """
    Generate actions continuously with FIXED capacity KV cache.
    When cache is full, MUST evict before adding new KV.
    """
    vla_model.eval()
    
    # Initialize KV cache with FIXED capacity
    kv_cache = {"semantic_keys": [], "traj_values": []}
    MAX_CAPACITY = 100  # Fixed maximum - cannot exceed
    
    generated_actions = []
    
    for timestep in range(len(vision_stream)):
        current_vision = vision_stream[timestep]
        
        # Build trajectory slice from recent actions
        if len(generated_actions) >= 10:
            recent_slice = torch.stack(generated_actions[-10:])
        elif len(generated_actions) > 0:
            recent_slice = torch.stack(generated_actions)
        else:
            recent_slice = None
        
        # Generate NEW KV pair if we have history
        if recent_slice is not None:
            with torch.no_grad():
                traj_latent = trajectory_compressor(recent_slice.unsqueeze(0))
                semantic_latent = llm_backbone(traj_latent)
                
                # CHECK CAPACITY BEFORE ADDING
                if len(kv_cache["semantic_keys"]) >= MAX_CAPACITY:
                    # MUST EVICT - cache is full
                    # Strategy 1: FIFO (remove oldest)
                    kv_cache["semantic_keys"].pop(0)
                    kv_cache["traj_values"].pop(0)
                    
                    # Strategy 2 (alternative): Attention-score based
                    # eviction_idx = find_lowest_attention_kv(kv_cache, intermediate_latent)
                    # kv_cache["semantic_keys"].pop(eviction_idx)
                    # kv_cache["traj_values"].pop(eviction_idx)
                    
                    # Strategy 3 (alternative): Critic-value based
                    # eviction_idx = find_lowest_value_kv(kv_cache, critic_head)
                    # kv_cache["semantic_keys"].pop(eviction_idx)
                    # kv_cache["traj_values"].pop(eviction_idx)
                
                # Append new KV (now there's space)
                kv_cache["semantic_keys"].append(semantic_latent)
                kv_cache["traj_values"].append(traj_latent)
                
                # Ensure we never exceed capacity
                assert len(kv_cache["semantic_keys"]) <= MAX_CAPACITY
        
        # Forward pass through VLA
        with torch.no_grad():
            outputs = vla_model(current_vision.unsqueeze(0), text, None)
            intermediate_latent = outputs["intermediate_latents"]
            
            # Retrieve from cache (always <= MAX_CAPACITY entries)
            if len(kv_cache["semantic_keys"]) > 0:
                mem_keys = torch.cat(kv_cache["semantic_keys"])
                mem_values = torch.cat(kv_cache["traj_values"])
                
                # Inference: NO causal mask needed
                # Already generating step-by-step, no future information exists
                retrieved = cross_attention(
                    query=intermediate_latent,
                    keys=mem_keys,
                    values=mem_values,
                    causal_mask=False  # Not needed in inference
                )
            else:
                retrieved = torch.zeros_like(intermediate_latent)
            
            # Generate action with memory fusion
            action = vla_model.action_head(retrieved)
        
        generated_actions.append(action.squeeze(0))
    
    return torch.stack(generated_actions)


def inference_with_episodes(vision_stream, text):
    """
    Training-style inference with explicit episode boundaries (for evaluation).
    """
    vla_model.eval()
    
    # Episode-based: clear cache at episode start
    for episode_data in vision_stream:
        # Clear cache for new episode
        kv_cache = {"semantic_keys": [], "traj_values": []}
        
        generated_actions = []
        
        for timestep, current_vision in enumerate(episode_data):
            # Build slice from recent actions
            if len(generated_actions) >= 10:
                recent_slice = torch.stack(generated_actions[-10:])
                
                with torch.no_grad():
                    traj_latent = trajectory_compressor(recent_slice.unsqueeze(0))
                    semantic_latent = llm_backbone(traj_latent)
                    
                    # Append to cache (no eviction needed in bounded episodes)
                    kv_cache["semantic_keys"].append(semantic_latent)
                    kv_cache["traj_values"].append(traj_latent)
            
            # Forward pass with cached KVs
            with torch.no_grad():
                outputs = vla_model(current_vision.unsqueeze(0), text, None)
                intermediate_latent = outputs["intermediate_latents"]
                
                if len(kv_cache["semantic_keys"]) > 0:
                    mem_keys = torch.cat(kv_cache["semantic_keys"])
                    mem_values = torch.cat(kv_cache["traj_values"])
                    retrieved = cross_attention(
                        query=intermediate_latent,
                        keys=mem_keys,
                        values=mem_values
                    )
                else:
                    retrieved = torch.zeros_like(intermediate_latent)
                
                action = vla_model.action_head(retrieved)
            
            generated_actions.append(action.squeeze(0))
        
        # Episode ends, cache cleared for next episode
        yield torch.stack(generated_actions)
```

---

## **5. Future Considerations**

---

### **5.1 Contrastive Alignment**

Add contrastive losses to align:

```
Semantic Latent ↔ Text Embedding (task description)
```

This improves grounding:

* "pick up cup"
* "open drawer"
* "push block to target"

---

### **5.2 Critic Supervision Implementation (TODO)**

Currently, the Critic architecture and loss function (`compute_critic_loss`) are implemented in `memory_module.py` but not integrated into the main training loop.

**Design Decision**: The Critic is a **lightweight MLP head** that operates on Semantic Latents (already processed by the LLM backbone), not a separate VLM instance.

**Required Steps for Implementation:**

1.  **Lightweight Critic Architecture**:
    *   Implement as a simple MLP: `Linear → LayerNorm → SiLU → Linear → Value`
    *   Input: Semantic Latent from LLM (B, 1, D_llm)
    *   Output: Critic Value (B, 1)
    *   This is much more efficient than using a separate VLM instance

2.  **Data Source**: Ensure the Dataset/DataLoader yields a `reward` or `success` signal (Ground Truth).
    *   This can be sourced from dataset metadata (e.g., `next.reward` or `success` fields in Parquet files).
    *   For expert demonstrations, `target_value` can default to `1.0`.

3.  **Training Loop Integration**:
    *   **Stage 1**: Train Critic head on Semantic Latents with MSE loss
    *   **Stage 2**: Freeze Critic, use it to compute Advantage for VLA training

4.  **Optimization**:
    *   Ensure gradients from `loss_critic` flow back through the Critic MLP to shape its weights for accurate value prediction.

---

Currently, the memory retrieval uses **standard cross-attention**. The full **Gated Cross-Attention** mechanism is planned for implementation.

**Planned Architecture**:

```python
class MemoryReadoutBlock(nn.Module):
    """
    Gated Cross-Attention for memory retrieval.
    TO BE IMPLEMENTED based on NeurIPS 2025 techniques.
    """
    def __init__(self, d_model, d_llm, num_heads):
        super().__init__()
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Zero-init gating
        self.gate = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    
    def forward(self, query, memory_keys, memory_values):
        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=query,
            key=memory_keys,
            value=memory_values
        )
        
        # Residual + Norm
        x = self.norm1(query + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        # Gating (zero-init)
        gate_value = torch.sigmoid(self.gate(x))  # (B, 1, 1)
        output = gate_value * x
        
        return output
```

**Benefits of Gating**:
- Memory starts with zero influence (gate ≈ 0)
- Model learns when to rely on memory vs. immediate inputs
- Prevents unstable training from uninitialized memory retrieval

**Current Workaround**: 
Until gated attention is implemented, use standard cross-attention with careful initialization and learning rate tuning.

Currently, the Critic architecture and loss function (`compute_critic_loss`) are implemented in `memory_module.py` but not integrated into the main training loop.

**Design Decision**: The Critic is a **lightweight MLP head** that operates on Semantic Latents (already processed by the LLM backbone), not a separate VLM instance.

**Required Steps for Implementation:**

1.  **Lightweight Critic Architecture**:
    *   Implement as a simple MLP: `Linear → LayerNorm → SiLU → Linear → Value`
    *   Input: Semantic Latent from LLM (B, 1, D_llm)
    *   Output: Critic Value (B, 1)
    *   This is much more efficient than using a separate VLM instance

2.  **Data Source**: Ensure the Dataset/DataLoader yields a `reward` or `success` signal (Ground Truth).
    *   This can be sourced from dataset metadata (e.g., `next.reward` or `success` fields in Parquet files).
    *   For expert demonstrations, `target_value` can default to `1.0`.

3.  **Training Loop Integration**:
    *   **Stage 1**: Train Critic head on Semantic Latents with MSE loss
    *   **Stage 2**: Freeze Critic, use it to compute Advantage for VLA training

4.  **Optimization**:
    *   Ensure gradients from `loss_critic` flow back through the Critic MLP to shape its weights for accurate value prediction.

```
[Stage 1: Train Critic & Build Memory Banks]

During this stage, the Memory Bank is being populated:

Ground Truth Actions (B, T, D_action)
      ↓ ActionEncoder
Raw Latents (B, T, D)
      ↓ TrajectoryCompressor
Trajectory Latent (B, 1, D_hidden)
      ↓ Projector → LLM (Backbone A)
Semantic Latent (B, 1, D_llm)
      ↓
      ├──→ Store in Memory Bank (Key: Semantic, Value: Trajectory)
      │
      └──→ Lightweight Critic MLP
                  ↓
          Critic Value (V_pred) [computed from Semantic Latent]
                  ↕
          Ground Truth (V_target)
                  ↓
          L_critic = MSE(V_pred, V_target)
                  ↓
          Backprop (Trains Critic MLP Only)

---

[Stage 2: Train VLA with Frozen Critic]

Vision + Text → Backbone → Action Head (Diffusion)
                                ↓
                    Intermediate Latent (Query) ──┐
                                                  │
Memory Bank:                                      │
  Keys: All Semantic Latents ◄────────────────────┤
  Values: All Trajectory Latents                  │
                                                  │
                                                  ▼
                                        Cross-Attention
                                                  ↓
                                    Fused Memory Context
                                                  ↓
                    Action Head (continues denoising)
                                                  ↓
                                    Predicted Actions
                                                  ↓
                        L_action_base = MSE(Predicted, GT)
                                                  ↓
                        [Compute Advantage from Frozen Critic]
                        Semantic Latent → Frozen_Critic_MLP → V_pred  [no grad]
                        Advantage (A) = V_target - V_pred
                                                  ↓
                        L_total = A · L_action_base
                                                  ↓
                        Backprop (Trains VLA Only)

---

Two-Stage Training Summary:
Stage 1: L_critic = MSE(V_pred, V_target)  →  Trains Lightweight Critic MLP
Stage 2: L_total = A · L_action             →  Trains VLA (Critic frozen, Advantage computed from Semantic Latent)

Note: Critic is a lightweight MLP operating on Semantic Latents (already processed by LLM).
      No separate VLM instance needed.
```
