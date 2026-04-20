# ST-GAT Architecture Report — After Gaze Integration & Bug Fixes

## What Changed (Summary)

| Category | Changes |
|---|---|
| **Bug fixes** | 2 critical runtime bugs in `train.py` |
| **New feature** | L2CS-Net gaze detection → 6-dim gaze features + directed gaze edges |
| **Hyperparameters** | 6 tuning changes for better convergence on 17:1 imbalanced data |
| **Files modified** | `config.py`, `model.py`, `train.py`, `preprocess.py`, `dataset.py` |

---

## 1. Bug Fixes

### Bug 1 — Validation AMP Crash (`train.py:182`)

```diff
- with autocast():                    # NameError: autocast not imported
+ with torch.amp.autocast('cuda'):    # matches train_epoch() at line 122
```

`autocast()` was never imported in `train.py`. This caused a `NameError` every time validation ran with AMP enabled (`use_amp=True`, the default). All validation metrics were silently broken.

### Bug 2 — Missing Argument in `evaluate()` (`train.py:637`)

```diff
- person_metrics = compute_person_metrics(all_probs, all_labels, all_person_ids)
+ person_metrics = compute_person_metrics(all_probs, all_labels, all_person_ids, all_clip_uids)
```

`compute_person_metrics()` requires 4 positional arguments: `probs`, `labels`, `person_ids`, `clip_uids`. The old code only passed 3, causing the `aggregate` parameter to receive a numpy array instead of `"mean"`. The evaluation loop now collects `clip_uids` per batch (same technique already used in `validate_epoch()`).

---

## 2. New Architecture

### 2.1 High-Level Overview

```
                        ┌─────────────────────────────────┐
                        │       Per-Clip Graph             │
                        │                                  │
                        │   Nodes: (person, frame) pairs   │
                        │                                  │
                        │   Edges:                         │
                        │     ─── Spatial (same frame)     │
                        │     ─── Temporal (consecutive)   │
                        │     ─── Temporal-Skip (long)     │
                        │     ─── Gaze (directed) ← NEW   │
                        └───────────────┬─────────────────┘
                                        │
                                        ▼
┌──────────────┐    ┌───────────────────────────────────────────┐
│ Node Features │    │           ST-GAT Model                     │
│               │    │                                            │
│ Face:   2048  │    │  Input Proj (2100→256)                     │
│ Audio:    40  │    │       │                                    │
│ Bbox:      6  │    │  ┌────▼────┐                              │
│ Gaze:      6  │──▶│  │ GATv2   │×4 layers (was 3)            │
│ ──────────── │    │  │ + Edge   │  4 heads, edge_dim=64       │
│ Total:  2100  │    │  │ Type Emb │  4 edge types (was 3)      │
│  (was 2094)   │    │  │ + Resid  │                             │
│               │    │  └────┬────┘                              │
└──────────────┘    │       │                                    │
                    │  ┌────▼────┐                               │
                    │  │   MLP    │  256→128→64→1                │
                    │  │Classifier│                               │
                    │  └────┬────┘                               │
                    │       │                                    │
                    │    logits (N,) per node                    │
                    └───────────────────────────────────────────┘
```

### 2.2 New Node Features — Gaze Vector (6-dim)

Previously each node had **2094** features (full mode). Now it has **2100**:

| Component | Dimensions | Source |
|---|---|---|
| Face embedding | 2048 | ResNet-50 (frozen, ImageNet-pretrained) |
| Audio MFCC | 40 | `librosa` per-frame MFCC |
| Bbox metadata | 6 | `[cx, cy, w, h, aspect, area]` normalized |
| **Gaze vector** | **6** | **L2CS-Net pitch/yaw + trigonometric encoding** |
| **Total** | **2100** | |

The 6-dim gaze vector is:

```
[pitch, yaw, sin(pitch), cos(pitch), sin(yaw), cos(yaw)]
```

- `pitch` and `yaw` are raw angles (radians) from L2CS-Net
- Trigonometric encoding provides smooth, continuous angular representation that avoids discontinuities at ±π boundaries

**Lite mode** (no video): node features go from 6 → **12** (bbox + zero gaze padding).

### 2.3 New Edge Type — Directed Gaze Edges

The graph now has **4 edge types** (was 3):

| Index | Type | Direction | Connection Rule |
|---|---|---|---|
| 0 | `EDGE_SPATIAL` | Bidirectional | Persons in same frame with bbox centers < 400px apart |
| 1 | `EDGE_TEMPORAL` | Bidirectional | Same person in consecutive frames (stride ≤ 1). Also self-loops |
| 2 | `EDGE_TEMPORAL_SKIP` | Bidirectional | Same person across frames with gap in (1, 3] (was 2) |
| **3** | **`EDGE_GAZE`** | **Directed** | **Person A → B if A's gaze yaw points within 0.4 rad (~23°) of the angle from A's bbox center toward B's bbox center** |

#### Gaze Edge Construction Logic

For each pair of persons (A, B) in the same frame:

```python
angle_A_to_B = atan2(center_B.y - center_A.y, center_B.x - center_A.x)
if |yaw_A - angle_A_to_B| < 0.4:      # A is looking toward B
    add directed edge A → B             # type = EDGE_GAZE

angle_B_to_A = atan2(center_A.y - center_B.y, center_A.x - center_B.x)  
if |yaw_B - angle_B_to_A| < 0.4:      # B is looking toward A
    add directed edge B → A             # type = EDGE_GAZE
```

**Why this matters for TTM:** If person A is looking directly at the camera wearer (whose position correlates with the image center), the gaze edge encodes that directional attention. The GATv2 attention mechanism learns to weight these gaze edges differently from spatial/temporal edges via the 4-type edge embedding.

### 2.4 Edge Type Embedding

```python
EdgeTypeEmbedding:
    nn.Embedding(4, 64)     # was Embedding(3, 64)
```

Each edge type gets a learnable 64-dim vector that is passed as `edge_attr` to `GATv2Conv`. The model learns distinct attention patterns for spatial proximity, temporal continuity, long-range temporal context, and gaze direction.

---

## 3. Preprocessing Pipeline (Gaze Extraction)

```
preprocess_split()
├── FaceFeatureExtractor (ResNet-50)      — existing
├── GazePipeline (L2CS-Net, ResNet50)     — NEW
│
└── for each clip:
    └── for each (person, frame):
        ├── crop face from video frame     — existing
        ├── ResNet-50 → 2048-dim           — existing
        ├── L2CS-Net → pitch, yaw          — NEW
        │   └── encode → [p, y, sin(p), cos(p), sin(y), cos(y)]
        └── bbox → 6-dim                   — existing
```

- **L2CS-Net** is initialized once per `preprocess_split()` call (not per clip)
- Wrapped in `try/except` — if face detection fails for a crop, falls back to `zeros(6)`
- If `l2cs` package is not installed, all gaze features are `zeros(6)` (graceful degradation)
- Stored in pkl as `gaze_features: {(pid, frame): ndarray(6,)}`

---

## 4. Hyperparameter Changes

| Parameter | Before | After | Why |
|---|---|---|---|
| `use_focal_loss` | `False` | `True` | Focal loss is purpose-built for extreme class imbalance; BCE with pos_weight alone was insufficient at 0.07 mAP |
| `focal_alpha` | 0.80 | 0.25 | Standard α from Lin et al. 2017 focal loss paper; γ=2.0 does the heavy lifting |
| `oversample_ratio` | 3.0 | 5.0 | More exposure to rare positive clips during training |
| `num_gat_layers` | 3 | 4 | Deeper message passing captures longer-range graph interactions, especially important with the new gaze edges |
| `patience` | 15 | 25 | More epochs before early stopping — needed since the model is now larger and has more to learn |
| `temporal_skip` | 2 | 3 | Wider temporal receptive field per person (connects frames up to 3 apart instead of 2) |

---

## 5. Model Parameter Count

With the new architecture (full mode, `node_input_dim=2100`, 4 GAT layers, 4 edge types):

| Component | Parameters (approx) |
|---|---|
| `input_proj` (Linear 2100→256 + LayerNorm) | ~538K |
| `edge_embed` (Embedding 4×64) | 256 |
| `gat_layers` ×4 (GATv2Conv + LayerNorm + residual) | ~1.1M |
| `classifier` (256→128→64→1 MLP) | ~41K |
| **Total trainable** | **~1.7M** |

---

## 6. Complete Data Flow

```
Video + Annotations
        │
        ▼
┌─────────────────────────────────────┐
│         PREPROCESSING                │
│  ResNet-50 face → 2048-dim           │
│  L2CS-Net gaze → [p, y, sin, cos]   │
│  librosa MFCC → 40-dim              │
│  bbox normalize → 6-dim             │
│  Output: .pkl per clip               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         GRAPH CONSTRUCTION           │
│  Nodes: (person, frame) = 2100-dim   │
│  Spatial edges (type 0)              │
│  Temporal edges (type 1)             │
│  Temporal-skip edges (type 2)        │
│  Gaze edges (type 3, directed)       │
│  Self-loops (type 1)                 │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         ST-GAT MODEL                 │
│  Input projection: 2100 → 256        │
│  Edge embedding: 4 types → 64-dim    │
│  GATv2 × 4 layers (4 heads each)     │
│  MLP classifier → per-node logit     │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         LOSS & METRICS               │
│  Focal Loss (α=0.25, γ=2.0)         │
│    + pos_weight=17.51 (capped 50)    │
│  Metrics: mAP, AUC-ROC, F1          │
│  Early stop on val node mAP          │
│  Best model → best_model.pt         │
└─────────────────────────────────────┘
```

---

## 7. Before You Train

```bash
# 1. Install gaze detection
pip install l2cs

# 2. Download weights
mkdir -p models
wget https://github.com/Ahmednull/L2CS-Net/releases/download/v0.1/L2CSNet_gaze360.pkl \
     -O models/L2CSNet_gaze360.pkl

# 3. Delete old preprocessed data (lacks gaze features)
rm -rf /DATA/DL_21/riceu/preprocessed_features/train/*.pkl
rm -rf /DATA/DL_21/riceu/preprocessed_features/val/*.pkl

# 4. Re-preprocess with gaze
python main.py preprocess --split both --mode full

# 5. Train
python main.py train --mode full --epochs 100

# 6. Evaluate
python main.py evaluate
```

---

## 8. Files Modified

| File | Lines Changed | What |
|---|---|---|
| `config.py` | +6 new fields, 6 value changes, `__post_init__` update | Gaze config + hyperparameter tuning |
| `model.py` | 2 lines | `Embedding(3→4)` for edge types |
| `train.py` | 4 lines | AMP bug fix + clip_uids bug fix |
| `preprocess.py` | ~40 lines added | L2CS-Net gaze extraction pipeline |
| `dataset.py` | ~50 lines added | Gaze features in node vectors + gaze edge construction |
