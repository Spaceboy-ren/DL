"""
Configuration for Spatio-Temporal GNN TTM Pipeline.
All paths, hyperparameters, and feature dimensions are defined here.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ──────────────────────────────────────────────────────────────
    # Data paths  (matched to lab PC layout)
    # ──────────────────────────────────────────────────────────────
    data_root: str = "/DATA/DL_21/wttm/ego4d_data/v2"
    annotation_dir: str = "/DATA/DL_21/wttm/ego4d_data/v2/annotations"
    clips_dir: str = "/DATA/DL_21/wttm/ego4d_data/v2/clips"          # {clip_uid}.mp4
    # Fallback: if clips aren't pre-cut, try full videos
    videos_dir: str = "/DATA/DL_21/wttm/ego4d_data/v2/full_scale"    # {video_uid}.mp4

    train_annotation: str = "ttm_train_clean.json"
    val_annotation: str = "ttm_val_clean.json"

    # Where preprocessed features get saved
    feature_dir: str = "preprocessed_features"

    # ──────────────────────────────────────────────────────────────
    # Preprocessing
    # ──────────────────────────────────────────────────────────────
    face_size: int = 224          # resize face crops to this before ResNet
    n_mfcc: int = 40              # number of MFCC coefficients
    audio_sr: int = 16000         # audio sample rate
    bbox_format: str = "xywh"     # "xywh" = top-left x, y, width, height
    bbox_pad_ratio: float = 0.15  # expand face bbox by this ratio for context

    # ──────────────────────────────────────────────────────────────
    # Graph construction
    # ──────────────────────────────────────────────────────────────
    spatial_threshold: float = 400.0   # pixel dist for spatial edges
    temporal_stride: int = 3           # connect frame t to t ± stride
    temporal_skip: int = 6             # also connect t to t ± skip (long range)
    frame_sample_rate: int = 3         # 10 FPS (keeps all temporal context without hitting the 24GB VRAM ceiling)
    min_frames_per_clip: int = 5       # skip clips with fewer frames

    # Edge type indices
    EDGE_SPATIAL: int = 0
    EDGE_TEMPORAL: int = 1
    EDGE_TEMPORAL_SKIP: int = 2
    EDGE_GAZE: int = 3
    gaze_feat_dim: int = 6
    gaze_model_path: str = "models/L2CSNet_gaze360.pkl"

    # ──────────────────────────────────────────────────────────────
    # Model architecture
    # ──────────────────────────────────────────────────────────────
    face_feat_dim: int = 2048     # ResNet-50 avgpool output
    audio_feat_dim: int = 40      # n_mfcc
    bbox_feat_dim: int = 6        # normalized bbox features
    node_input_dim: int = 0       # auto-calculated
    hidden_dim: int = 256
    num_gat_layers: int = 4
    num_heads: int = 4
    gat_dropout: float = 0.2
    classifier_dropout: float = 0.4
    use_edge_type: bool = True    # encode edge types in GAT

    # ──────────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────────
    batch_size: int = 2
    lr: float = 5e-4
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 10           # early stopping patience
    warmup_epochs: int = 3

    # Class imbalance handling
    focal_alpha: float = 0.25     # standard for imbalanced binary classification
    focal_gamma: float = 2.0      # focusing parameter
    pos_weight: float = 17.51     # Hardcoded to bypass the 1.5-hour scan (4778407 neg / 272901 pos)
    use_focal_loss: bool = True   # Focal loss handles 17:1 imbalance better
    oversample_positive: bool = True
    oversample_ratio: float = 5.0  # repeat positive clips this many times

    grad_clip: float = 1.0
    use_amp: bool = True           # automatic mixed precision

    # ──────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────
    eval_aggregate: str = "mean"   # "mean" or "max" to aggregate per-person
    tta: bool = False              # test-time augmentation

    # ──────────────────────────────────────────────────────────────
    # Misc
    # ──────────────────────────────────────────────────────────────
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    experiment_name: str = "stgnn_ttm_v1"

    # ──────────────────────────────────────────────────────────────
    # Feature extraction mode
    # ──────────────────────────────────────────────────────────────
    # "full"    = ResNet-50 visual + MFCC audio  (requires video files)
    # "lite"    = bbox metadata only             (no video files needed)
    feature_mode: str = "full"

    def __post_init__(self):
        """Auto-calculate derived fields and create directories."""
        if self.feature_mode == "full":
            self.node_input_dim = (self.face_feat_dim + self.audio_feat_dim +
                                   self.bbox_feat_dim + self.gaze_feat_dim)
        else:
            self.node_input_dim = self.bbox_feat_dim + self.gaze_feat_dim

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

    @property
    def train_json_path(self) -> str:
        return os.path.join(self.annotation_dir, self.train_annotation)

    @property
    def val_json_path(self) -> str:
        return os.path.join(self.annotation_dir, self.val_annotation)
