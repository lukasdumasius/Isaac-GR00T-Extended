## Implementation Status (Dec 2025)

The only code change from the original feature fusion design is the addition of a dummy term to all intermediate features in the EagleBackbone for DDP/gradient connectivity. No attention mask is applied in DiT cross-attention (encoder_attention_mask=None), matching the baseline. Hierarchical routing, proportional slicing, and block-feature mapping remain as described above.

## Troubleshooting

If you see loss = 0 or gradients are NaN, check that the dummy term is present in all intermediate features. If problems persist, try adding proper attention masking for intermediate features in the DiT blocks.
# Intermediate Feature Fusion

This implementation adds hierarchical routing of intermediate Eagle-2 VLM features to DiT blocks, with three different fusion modes.

## Overview

Instead of using only the final Eagle-2 output for cross-attention, intermediate layers from the VLM are extracted and routed to DiT blocks. The fusion mode determines how these features are combined:

1. **Per-Layer (Simple)**: Each DiT block receives a distinct Eagle layer (hierarchically mapped)
2. **Per-Layer (Full)**: Same as simple, but with per-layer learnable processing
3. **Global**: All Eagle layers are fused into one feature that all DiT blocks receive

## Configuration

Enable and configure in your config:

```python
"backbone_config": {
    "extract_intermediate_layers": True,
    "num_intermediate_layers": 4,  # Recommended range: 4-12 (default: 4)
    "intermediate_feature_fusion_mode": "per_layer_feature_full",  # Choose mode (default)
}

"action_head_config": {
    "intermediate_feature_fusion_mode": "per_layer_feature_full",  # Must match backbone
    "num_intermediate_layers": 4,  # Must match backbone
}
```

- **num_intermediate_layers**: Evenly-spaced layers to extract from Eagle-2 (32 layers available). DiT has 12 blocks, so 4-12 recommended (default: 4).
- **intermediate_feature_fusion_mode**: Controls how features are processed.

### Modes

| Aspect | Per-Layer Simple | Per-Layer Full | Simplified Global Feature |
|--------|-----------------|----------------|---------------------------|
| Feature routing | Hierarchical | Hierarchical | Single feature to all blocks |
| Eagle projection | Shared linear | Per-layer linear | Shared linear + fusion |
| DiT processing | Shared LayerNorm + self-attention | Per-layer LayerNorm + self-attention | Shared LayerNorm + self-attention |
| Parameters | Minimal | Maximum | Medium |
| Complexity | Low | High | Low |

- **per_layer_feature_full** (default, recommended)
  - Early Eagle layers feed early DiT blocks, late Eagle layers feed late blocks
  - Per-layer linear projections in eagle_backbone (one per extracted layer)
  - Per-layer LayerNorm and self-attention in flow_matching_action_head

- **per_layer_feature_simple**
  - Same hierarchical routing as full mode
  - Shared linear projection (2048 → 1536 dims) in eagle_backbone
  - Shared LayerNorm and self-attention in flow_matching_action_head

- **simplified_global_feature**
  - All intermediate Eagle layers averaged into one global feature
  - All DiT blocks receive this single global feature
  - Shared LayerNorm and self-attention applied to the global feature
  - Hybrid approach combining baseline simplicity with intermediate feature richness

## Architecture Details

- **Eagle-2**: 32 layers, 2048-dim outputs
- **DiT**: 12 blocks for per-layer modes; receives 1 feature for global mode
- **Recommended num_intermediate_layers**: 4-12 (matching or exceeding DiT blocks for per-layer modes)

## Files Modified

- `gr00t/model/backbone/eagle_backbone.py` - Feature extraction, projection, fusion, and DDP dummy term logic
- `gr00t/model/action_head/flow_matching_action_head.py` - Feature processing pipeline
- `gr00t/model/action_head/cross_attention_dit.py` - Hierarchical routing (no logic changes except DDP dummy term support)

## Backward Compatibility

Disabled by default (`extract_intermediate_layers: False`). Existing configs work unchanged.

To completely disable intermediate feature fusion and run gr00t in its original design, set:

```python
"backbone_config": {
    "extract_intermediate_layers": False,  # Disables all intermediate feature extraction
}
```

When disabled, gr00t runs with zero overhead from intermediate features—it operates exactly as originally designed.
