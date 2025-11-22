# Architecture Diagrams: Intermediate Features Implementation

## 1. Original Architecture 

```
┌─────────────────────────────────────────────────────────────────┐
│                         Eagle-2 VLM                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         ┌──────────┐ │
│  │ Layer  0 │→ │ Layer  1 │→ │ Layer  2 │ ... →  │ Layer N  │ │
│  └──────────┘  └──────────┘  └──────────┘         └──────────┘ │
│                                                           ↓      │
│                                                    [Final Features]
│                                                    [1536-dim]   │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
                        ┌─────────────────────┐
                        │   Normalize &       │
                        │   Self-Attention    │
                        └─────────────────────┘
                                  ↓
        ┌─────────────────────────────────────────────────────────┐
        │                    DiT Transformer                       │
        │  ┌─────────┐ ┌─────────┐ ┌─────────┐   ┌─────────┐    │
        │  │ Block 0 │ │ Block 1 │ │ Block 2 │...│Block 11 │    │
        │  │  ↓      │ │  ↓      │ │  ↓      │   │  ↓      │    │
        │  │[Same]   │ │[Same]   │ │[Same]   │...|[Same]   │    │
        │  │Features │ │Features │ │Features │   │Features │    │
        │  └─────────┘ └─────────┘ └─────────┘   └─────────┘    │
        │                                                          │
        │              All blocks use IDENTICAL features          │
        └─────────────────────────────────────────────────────────┘
                                  ↓
                        ┌─────────────────────┐
                        │  Action Prediction  │
                        │  (diffusion model)  │
                        └─────────────────────┘
```

**Problem**: Limited diversity - each DiT layer has same information

---

## 2. New Architecture (After Modification)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Eagle-2 VLM                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         ┌──────────┐ │
│  │ Layer  0 │→ │ Layer  1 │→ │ Layer  2 │ ... →  │ Layer N  │ │
│  └──────────┘  └──────────┘  └──────────┘         └──────────┘ │
│                    ↓                ↓                      ↓     │
│        [Early]  [Mid-Early]   [Mid-Late]           [Final]      │
│         ↓          ↓              ↓                  ↓          │
│      Extract at evenly-spaced intervals             │          │
│         ↓          ↓              ↓                  ↓          │
│    [Features_A] [Features_B] [Features_C]    [Features_D]      │
│    [1536-dim]   [1536-dim]   [1536-dim]      [1536-dim]       │
└─────────────────────────────────────────────────────────────────┘
         ↓              ↓              ↓                  ↓
    ┌────────┐    ┌────────┐    ┌────────┐        ┌────────┐
    │Norm &  │    │Norm &  │    │Norm &  │        │Norm &  │
    │Self-   │    │Self-   │    │Self-   │        │Self-   │
    │Attn    │    │Attn    │    │Attn    │        │Attn    │
    └────────┘    └────────┘    └────────┘        └────────┘
         ↓              ↓              ↓                  ↓
┌───────────────────────────────────────────────────────────────────┐
│                    DiT Transformer (12 blocks)                    │
│                                                                   │
│  Features_A  Features_A  Features_A  Features_A                  │
│      ↓          ↓          ↓          ↓                          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐                         │
│  │Block │  │Block │  │Block │  │Block │                         │
│  │  0   │  │  1   │  │  2   │  │  3   │                         │
│  │(CA)  │  │(CA)  │  │(CA)  │  │(CA)  │    [Early Features]    │
│  └──────┘  └──────┘  └──────┘  └──────┘                         │
│                                                                   │
│              Features_B  Features_B  Features_B                   │
│                   ↓          ↓          ↓                         │
│              ┌──────┐  ┌──────┐  ┌──────┐                        │
│              │Block │  │Block │  │Block │                        │
│              │  4   │  │  5   │  │  6   │                        │
│              │(CA)  │  │(CA)  │  │(CA)  │   [Mid-Early Features]│
│              └──────┘  └──────┘  └──────┘                        │
│                                                                   │
│                   Features_C  Features_C                          │
│                        ↓          ↓                              │
│                   ┌──────┐  ┌──────┐                             │
│                   │Block │  │Block │                             │
│                   │  7   │  │  8   │                             │
│                   │(CA)  │  │(CA)  │    [Mid-Late Features]     │
│                   └──────┘  └──────┘                             │
│                                                                   │
│                            Features_D                             │
│                                 ↓                                │
│                            ┌──────┐  ┌──────┐                    │
│                            │Block │  │Block │                    │
│                            │  9   │  │ 10   │                    │
│                            │(CA)  │  │(CA)  │   [Final Features] │
│                            └──────┘  └──────┘                    │
│                                                                   │
│                            Features_D                             │
│                                 ↓                                │
│                            ┌──────┐                               │
│                            │Block │                               │
│                            │ 11   │                               │
│                            │(CA)  │                               │
│                            └──────┘                               │
│                                                                   │
│       (CA) = Cross-Attention Layer                               │
│  Each block receives DIFFERENT feature level                    │
└───────────────────────────────────────────────────────────────────┘
                                ↓
                      ┌─────────────────────┐
                      │  Action Prediction  │
                      │  (diffusion model)  │
                      └─────────────────────┘
```

**Benefit**: Rich, hierarchical conditioning with different semantic levels

---

## 3. Feature Extraction Process

```
Eagle-2 Transformer Stack (Example: 32 layers)

  Layer 0:    Initial embeddings
    ↓
  Layer 1-7:  Low-level patterns
    ↓
  Layer 8:    ◄─── Selected: Features_A (Early)
    ↓
  Layer 9-15: Intermediate representations
    ↓
  Layer 16:   ◄─── Selected: Features_B (Mid-Early)
    ↓
  Layer 17-23: High-level semantics
    ↓
  Layer 24:   ◄─── Selected: Features_C (Mid-Late)
    ↓
  Layer 25-31: Task-specific representations
    ↓
  Layer 32:   ◄─── Selected: Features_D (Final)

Selection Strategy: Even spacing across depth
  num_intermediate_layers = 4
  step = (32 - 0) / (4 - 1) ≈ 10
  layers_selected = [0, 10, 20, 32] → [8, 16, 24, 32]
```

---

## 4. Routing Mechanism

```
Given:
- N_FEATURES = 4 intermediate feature levels
- N_BLOCKS = 12 DiT transformer blocks
- Calculate: step = (4-1) / (12-1) ≈ 0.27

Block Assignment:
  Block  0: feature_idx = int(0 × 0.27) = 0 → Features_A
  Block  1: feature_idx = int(1 × 0.27) = 0 → Features_A
  Block  2: feature_idx = int(2 × 0.27) = 0 → Features_A
  Block  3: feature_idx = int(3 × 0.27) = 0 → Features_A
  Block  4: feature_idx = int(4 × 0.27) = 1 → Features_B
  Block  5: feature_idx = int(5 × 0.27) = 1 → Features_B
  Block  6: feature_idx = int(6 × 0.27) = 1 → Features_B
  Block  7: feature_idx = int(7 × 0.27) = 1 → Features_B
  Block  8: feature_idx = int(8 × 0.27) = 2 → Features_C
  Block  9: feature_idx = int(9 × 0.27) = 2 → Features_C
  Block 10: feature_idx = int(10 × 0.27) = 2 → Features_C
  Block 11: feature_idx = int(11 × 0.27) = 2 → Features_C

Result: Smooth mapping from early to late features
```

---

## 5. Data Flow Diagram

```
Training Forward Pass:

Input (images, text)
     ↓
┌────────────────────────────────┐
│   EagleBackbone.forward()      │
│  ┌────────────────────────────┐│
│  │Extract Intermediate Layers ││
│  │ [A, B, C, D] + attention  ││
│  └────────────────────────────┘│
└────────────┬───────────────────┘
             ↓
    ┌────────────────────────────┐
    │FlowmatchingActionHead      │
    │.forward()                  │
    │ ┌────────────────────────┐ │
    │ │process_backbone_output│ │ Normalize & process
    │ │ each intermediate     │ │ each feature level
    │ └────────────────────────┘ │
    └────────────┬───────────────┘
                 ↓
    ┌────────────────────────────┐
    │DiT.forward()               │
    │ ┌────────────────────────┐ │
    │ │hierarchical routing    │ │ Map features to blocks
    │ │[A→blocks 0-3           │ │
    │ │ B→blocks 4-7           │ │
    │ │ C→blocks 8-10          │ │
    │ │ D→blocks 11]           │ │
    │ └────────────────────────┘ │
    │ ┌────────────────────────┐ │
    │ │12 transformer blocks   │ │ Process with varied conditioning
    │ └────────────────────────┘ │
    └────────────┬───────────────┘
                 ↓
    Action Prediction (MSE loss)
                 ↓
            Backprop
```

---

## 6. Impact on Information Flow

```
Information Richness Over DiT Depth:

Without Intermediate Features:
  Block 0:  [████] Same features
  Block 1:  [████] Same features
  Block 2:  [████] Same features
  Block 3:  [████] Same features
  Block 4:  [████] Same features
  Block 5:  [████] Same features
  Block 6:  [████] Same features
  Block 7:  [████] Same features
  Block 8:  [████] Same features
  Block 9:  [████] Same features
  Block 10: [████] Same features
  Block 11: [████] Same features
  
  Diversity: Low - all blocks get identical input

With Intermediate Features:
  Block 0:  [█░░░] Early patterns
  Block 1:  [█░░░] Early patterns
  Block 2:  [█░░░] Early patterns
  Block 3:  [█░░░] Early patterns
  Block 4:  [██░░] Early + mid patterns
  Block 5:  [██░░] Early + mid patterns
  Block 6:  [██░░] Early + mid patterns
  Block 7:  [██░░] Early + mid patterns
  Block 8:  [███░] Mid + semantic patterns
  Block 9:  [███░] Mid + semantic patterns
  Block 10: [███░] Mid + semantic patterns
  Block 11: [████] Full semantic + task patterns
  
  Diversity: High - progressive abstraction levels
```

---

This architecture provides a natural, multi-scale conditioning pipeline that leverages the hierarchical nature of transformer models.
