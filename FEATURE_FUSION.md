# Intermediate Feature Fusion

This implementation adds hierarchical routing of intermediate Eagle-2 VLM features to DiT blocks.

## Overview

Instead of using only the final Eagle-2 output for cross-attention, intermediate layers from the VLM are now extracted and routed to corresponding DiT blocks. Early VLM layers feed early DiT blocks, late VLM layers feed late blocks.

## Configuration

Enable in your config:

```python
"backbone_config": {
    "extract_intermediate_layers": True,
    "num_intermediate_layers": 12,  # Recommended range: 4-12
    "simplified_feature_fusion": True,  # Use shared LayerNorm projection
}
```

- **num_intermediate_layers**: Number of evenly-spaced layers to extract from Eagle-2 (32 layers available). DiT has 12 blocks, so 4-12 is recommended (matching or exceeding the number of DiT blocks ensures each block receives a distinct feature).
- **simplified_feature_fusion**: When True (default), uses a shared LayerNorm projection for all intermediate features. Set to False to use per-layer normalization and attention (experimental).

### What Changes with simplified_feature_fusion

**Simplified Mode (True):**
- Eagle backbone uses one shared **linear projection** (2048 → 1536 dims)
- FlowmatchingActionHead applies one shared **LayerNorm** to all intermediate features

**Complex Mode (False):**
- Eagle backbone creates per-layer **linear projections** (one 2048 → 1536 for each intermediate layer)
- FlowmatchingActionHead applies per-layer **LayerNorm and self-attention** to each intermediate feature

## How It Works

1. **Feature Extraction**: Evenly-spaced layers selected from Eagle-2 (e.g., layers 8, 16, 24, 32 for 4 layers)
2. **Projection**: All features projected from 2048-dim to 1536-dim for consistency
3. **Hierarchical Routing**: Feature i maps to DiT block i based on depth ratio
4. **Processing**: Features optionally normalized before cross-attention

## Files Modified

- `gr00t/model/backbone/eagle_backbone.py` - Feature extraction and projection
- `gr00t/model/action_head/flow_matching_action_head.py` - Feature processing pipeline
- `gr00t/model/action_head/cross_attention_dit.py` - Hierarchical routing (no changes needed)

## Backward Compatibility

Disabled by default (`extract_intermediate_layers: False`). Existing configs work unchanged.
