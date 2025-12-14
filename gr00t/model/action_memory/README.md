# Action Memory Implementation (Simplified Version)

This directory contains the implementation of the simplified action memory mechanism for GR00T, where trajectory latents are directly used for memory retrieval without LLM processing.

## Overview

The action memory module enables the model to retrieve and condition on relevant past trajectory experiences during action generation. In the simplified version:
- **No LLM processing** of trajectory latents
- **Direct cross-attention** between DiT action latents (Query) and stored trajectory latents (Key/Value)
- **Memory at 4 DiT blocks** (3, 7, 10, 11) aligned with EAGLE2's hierarchical features

## Architecture

```
Vision + Text → EAGLE2 → DiT (12 blocks) → Actions
                             ↓
                        Cross-Attention (at blocks 3,7,10,11)
                             ↓
                        Memory Bank (trajectory latents)
```

## Key Components

### 1. `memory_cross_attention.py`

#### `MemoryCrossAttention`
Multi-head cross-attention module for retrieving memories:
- **Query**: Action latents from DiT blocks
- **Key**: Trajectory latents from memory bank
- **Value**: Same trajectory latents
- Supports gated attention (TODO)

#### `MemoryBank`
Manages trajectory latent storage:
- Fixed capacity with eviction strategies (FIFO, random)
- Per-episode cache during training
- Continuous cache during inference

#### `TrajectoryEncoder`
Encodes action sequences into compact trajectory latents:
- MLP or 1D Conv encoder
- Input: (B, window_size, D_action)
- Output: (B, 1, D_memory)

### 2. `cross_attention_dit.py` (Modified)

Added memory cross-attention support to DiT:
- New parameters: `enable_memory_cross_attention`, `memory_cross_attention_layers`, `memory_dim`
- Memory cross-attention modules at layers 3, 7, 10, 11
- Forward pass accepts `memory_keys` and `memory_values`

### 3. `flow_matching_action_head.py` (Modified)

Modified to pass memory to DiT:
- `forward()`: Added `memory_keys` and `memory_values` parameters
- `get_action()`: Same memory parameters for inference

### 4. `memory_training_utils.py`

#### `ActionMemoryTrainingWrapper`
Helper for building per-sample memory during training:
```python
wrapper = ActionMemoryTrainingWrapper(
    action_dim=7,
    memory_dim=512,
    trajectory_window=10,
)

memory_keys, memory_values = wrapper.build_per_sample_memory(
    batch_actions=batch_actions,  # (B, T_total, D_action)
)

output = model(
    backbone_output=backbone_output,
    action_input=action_input,
    memory_keys=memory_keys,
    memory_values=memory_values,
)
```

#### `ActionMemoryInferenceManager`
Manager for continuous memory during inference:
```python
manager = ActionMemoryInferenceManager(
    action_dim=7,
    memory_dim=512,
    max_memory_capacity=100,
)

# In rollout loop
for step in range(max_steps):
    memory_keys, memory_values = manager.get_memory()
    
    action = model.get_action(
        backbone_output=backbone_output,
        action_input=action_input,
        memory_keys=memory_keys,
        memory_values=memory_values,
    )
    
    manager.add_action(action)
```

## Usage

### Training

1. **Enable memory in DiT config**:
```python
diffusion_model_cfg = {
    "num_layers": 12,
    "enable_memory_cross_attention": True,
    "memory_cross_attention_layers": (3, 7, 10, 11),
    "memory_dim": 512,
    # ... other configs
}
```

2. **Create memory wrapper**:
```python
from gr00t.model.action_memory.memory_training_utils import ActionMemoryTrainingWrapper

memory_wrapper = ActionMemoryTrainingWrapper(
    action_dim=7,
    memory_dim=512,
    trajectory_window=10,
)
```

3. **In training loop**:
```python
for batch in dataloader:
    # Build per-sample memory
    memory_keys, memory_values = memory_wrapper.build_per_sample_memory(
        batch_actions=batch['actions'],  # (B, T_total, D_action)
    )
    
    # Forward with memory
    output = model(
        backbone_output=batch['backbone_output'],
        action_input=batch['action_input'],
        memory_keys=memory_keys,
        memory_values=memory_values,
    )
    
    loss = output['loss']
    loss.backward()
    optimizer.step()
```

### Inference

```python
from gr00t.model.action_memory.memory_training_utils import ActionMemoryInferenceManager

# Initialize manager
inference_manager = ActionMemoryInferenceManager(
    action_dim=7,
    memory_dim=512,
    trajectory_window=10,
    max_memory_capacity=100,
    eviction_strategy="fifo",
)

# Clear at episode start
inference_manager.clear()

# Rollout loop
for step in range(max_steps):
    # Get current memory
    memory_keys, memory_values = inference_manager.get_memory()
    
    # Get action
    action = model.get_action(
        backbone_output=backbone_output,
        action_input=action_input,
        memory_keys=memory_keys,
        memory_values=memory_values,
    )
    
    # Execute and add to memory
    obs, reward, done, info = env.step(action)
    inference_manager.add_action(action)
```

## Design Details

### Memory Cross-Attention Layers

We add cross-attention at 4 specific DiT blocks:
- **Block 3**: After early processing with EAGLE2 Layer 8 features
- **Block 7**: After mid-early processing with EAGLE2 Layer 16 features
- **Block 10**: After mid-late processing with EAGLE2 Layer 24 features
- **Block 11**: After final processing with EAGLE2 Layer 32 features

This matches EAGLE2's hierarchical feature extraction and provides rich, multi-level conditioning.

### Training vs Inference Memory

**Training:**
- Per-sample memory bank (each batch item has its own memory)
- Built from ground-truth action trajectories
- Cleared after each episode/batch
- No capacity limit (memory size = number of trajectory slices)

**Inference:**
- Single continuous memory bank
- Built incrementally from generated actions
- Fixed capacity with eviction when full
- Persists across steps until episode end

### Trajectory Slicing

Actions are sliced into overlapping windows:
- Window size: 10 actions (configurable)
- Overlap: 50% (stride = window_size // 2)
- Each slice is encoded into a single trajectory latent

## Future Work

- [ ] **Gated Cross-Attention** (NeurIPS 2025 target)
  - Add learnable gates to control memory influence
  - Zero-init gates for stability
  - Adaptive memory weighting per layer

- [ ] **Attention-based Eviction**
  - Evict memories with lowest attention scores
  - Keep most relevant memories

- [ ] **Critic-based Filtering** (Optional)
  - Add lightweight critic to score trajectory quality
  - Filter low-quality memories during inference

- [ ] **Memory Compression**
  - Compress old memories to save capacity
  - Hierarchical memory structure

## Files

- `memory_cross_attention.py`: Core memory modules
- `memory_training_utils.py`: Training and inference helpers
- `Action_Memory_Design.md`: Detailed architecture documentation
- `README.md`: This file

## References

See `Action_Memory_Design.md` for detailed architecture diagrams and mathematical formulations.

