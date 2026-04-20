"""
Preprocess Ego4D TTM data: extract face features (ResNet-50), audio MFCC,
and save per-clip feature bundles for fast graph dataset loading.

Usage:
    python preprocess.py --split train
    python preprocess.py --split val
    python preprocess.py --split train --mode lite   # bbox-only (no video needed)
"""

import argparse
import json
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path

# Suppress repetitive librosa fallback warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm import tqdm

try:
    from l2cs import Pipeline as GazePipeline
    HAS_L2CS = True
except ImportError:
    HAS_L2CS = False

from config import Config


# ─────────────────────────────────────────────────────────────────
# Feature extractor: ResNet-50
# ─────────────────────────────────────────────────────────────────

class FaceFeatureExtractor:
    """Extract 2048-dim face embeddings using ResNet-50 (ImageNet pretrained)."""

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load pretrained ResNet-50, remove final FC layer
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = nn.Sequential(*list(resnet.children())[:-1])  # output: (B, 2048, 1, 1)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_batch(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of face crops.

        Args:
            face_crops: list of BGR numpy arrays (H, W, 3)

        Returns:
            features: (N, 2048) numpy array
        """
        if len(face_crops) == 0:
            return np.zeros((0, 2048), dtype=np.float32)

        tensors = []
        for crop in face_crops:
            if crop is None or crop.size == 0:
                # Create a black image if crop is invalid
                crop = np.zeros((224, 224, 3), dtype=np.uint8)
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(crop_rgb))

        batch = torch.stack(tensors).to(self.device)
        features = self.model(batch).squeeze(-1).squeeze(-1)  # (N, 2048)
        return features.cpu().numpy()


# ─────────────────────────────────────────────────────────────────
# Audio feature extractor
# ─────────────────────────────────────────────────────────────────

def extract_audio_mfcc(video_path: str, sr: int = 16000, n_mfcc: int = 40,
                       fps: float = 30.0, num_frames: int = None) -> np.ndarray:
    """
    Extract per-frame MFCC features from video audio track.

    Returns:
        mfcc_per_frame: (num_frames, n_mfcc) numpy array
    """
    try:
        # Extract audio using librosa (reads audio from video files)
        y, _ = librosa.load(video_path, sr=sr, mono=True)

        if len(y) == 0:
            raise ValueError("Empty audio track")

        # Compute MFCC
        hop_length = int(sr / fps)  # one MFCC vector per video frame
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                     hop_length=hop_length)  # (n_mfcc, T)
        mfcc = mfcc.T  # (T, n_mfcc)

        # Pad or truncate to match num_frames
        if num_frames is not None:
            if mfcc.shape[0] < num_frames:
                pad = np.zeros((num_frames - mfcc.shape[0], n_mfcc), dtype=np.float32)
                mfcc = np.vstack([mfcc, pad])
            else:
                mfcc = mfcc[:num_frames]

        return mfcc.astype(np.float32)

    except Exception as e:
        print(f"    [WARN] Audio extraction failed for {video_path}: {e}")
        if num_frames is not None:
            return np.zeros((num_frames, n_mfcc), dtype=np.float32)
        return np.zeros((1, n_mfcc), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# Bbox utilities
# ─────────────────────────────────────────────────────────────────

def expand_bbox(x, y, w, h, pad_ratio: float, img_w: int, img_h: int):
    """Expand bbox by pad_ratio on each side, clamp to image bounds."""
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(img_w, int(x + w + pad_w))
    y2 = min(img_h, int(y + h + pad_h))
    return x1, y1, x2, y2


def compute_bbox_features(bbox, img_w: int = 1920, img_h: int = 1080) -> np.ndarray:
    """
    Compute normalized bbox features: [cx, cy, w, h, aspect_ratio, area_ratio].
    All values normalized to [0, 1] range.
    """
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    aspect = w / max(h, 1e-6)
    area = (w * h) / (img_w * img_h)
    return np.array([cx, cy, nw, nh, aspect, area], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# Video reader
# ─────────────────────────────────────────────────────────────────

def find_video_path(clip_uid: str, video_uid: str, cfg: Config) -> str | None:
    """Try to locate the video file for a given clip."""
    # Try clip-based path first
    candidates = [
        os.path.join(cfg.clips_dir, f"{clip_uid}.mp4"),
        os.path.join(cfg.clips_dir, clip_uid, f"{clip_uid}.mp4"),
        os.path.join(cfg.videos_dir, f"{video_uid}.mp4"),
        os.path.join(cfg.data_root, "clips", f"{clip_uid}.mp4"),
        os.path.join(cfg.data_root, "clips_hq", f"{clip_uid}.mp4"),
        os.path.join(cfg.data_root, "video_clips", f"{clip_uid}.mp4"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def read_video_frames(video_path: str, frame_indices: list[int]) -> dict[int, np.ndarray]:
    """
    Read specific frames from a video file.

    Returns:
        dict mapping frame_index -> BGR numpy array
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [WARN] Cannot open video: {video_path}")
        return {}

    frames = {}
    max_frame = max(frame_indices)

    frame_set = set(frame_indices)
    idx = 0
    while idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_set:
            frames[idx] = frame
        idx += 1

    cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────
# Main preprocessing
# ─────────────────────────────────────────────────────────────────

def load_annotations(json_path: str) -> dict:
    """
    Load TTM annotations and group by clip_uid.

    Returns:
        clip_data: dict[clip_uid] -> list of annotation entries
    """
    print(f"Loading annotations from {json_path} ...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle both list and dict formats
    if isinstance(data, dict):
        # Some versions wrap in a dict with a key
        for key in ["clips", "data", "annotations"]:
            if key in data:
                data = data[key]
                break
        if isinstance(data, dict):
            # Might be keyed by clip_uid already
            return data

    # Group by clip_uid
    clip_data = defaultdict(list)
    for entry in data:
        clip_data[entry["clip_uid"]].append(entry)

    print(f"  Found {len(clip_data)} clips, {len(data)} total entries")

    # Print class distribution
    labels = [e["ttm_label"] for e in data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Class distribution: TTM=0: {n_neg} ({n_neg/len(labels)*100:.1f}%), "
          f"TTM=1: {n_pos} ({n_pos/len(labels)*100:.1f}%)")

    return dict(clip_data)


def preprocess_clip_full(clip_uid: str, entries: list, cfg: Config,
                         face_extractor: FaceFeatureExtractor,
                         gaze_pipeline=None) -> dict | None:
    """
    Extract features for a single clip (full mode: visual + audio).

    Returns dict with:
        - face_features: dict[(person_id, frame)] -> (2048,) numpy array
        - audio_features: (num_frames, n_mfcc) numpy array
        - bbox_features: dict[(person_id, frame)] -> (6,) numpy array
        - labels: dict[(person_id, frame)] -> int
        - bboxes: dict[(person_id, frame)] -> (4,) [x, y, w, h]
        - metadata: dict with clip info
    """
    video_uid = entries[0]["video_uid"]
    video_path = find_video_path(clip_uid, video_uid, cfg)

    if video_path is None:
        return None

    # Organize entries by (person_id, frame)
    entry_map = {}
    frame_indices = set()
    person_ids = set()
    for e in entries:
        pid = str(e["person_id"])
        frame = int(e["frame"])
        entry_map[(pid, frame)] = e
        frame_indices.add(frame)
        person_ids.add(pid)

    frame_indices = sorted(frame_indices)
    person_ids = sorted(person_ids)

    if len(frame_indices) < cfg.min_frames_per_clip:
        return None

    # Apply frame_sample_rate (e.g. 10 = 3 FPS) to keep temporal story but reduce nodes
    if cfg.frame_sample_rate > 1:
        frame_indices = [f for f in frame_indices if f % cfg.frame_sample_rate == 0]
        # Filter entry_map
        frame_set = set(frame_indices)
        entry_map = {k: v for k, v in entry_map.items() if k[1] in frame_set}

    # Read video frames
    frames_dict = read_video_frames(video_path, frame_indices)
    if len(frames_dict) == 0:
        return None

    # Get video dimensions from first frame
    first_frame = next(iter(frames_dict.values()))
    img_h, img_w = first_frame.shape[:2]

    # Extract face crops and features
    face_features = {}
    bbox_features = {}
    labels = {}
    bboxes_out = {}

    # Collect face crops in batches for efficiency
    crop_keys = []
    crop_images = []
    gaze_features = {}

    for (pid, frame), entry in entry_map.items():
        if frame not in frames_dict:
            continue

        bbox = entry["bbox"]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        # Expand and clamp bbox
        x1, y1, x2, y2 = expand_bbox(x, y, w, h, cfg.bbox_pad_ratio, img_w, img_h)

        # Crop face
        img = frames_dict[frame]
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)

        crop_keys.append((pid, frame))
        crop_images.append(crop)

        # Bbox features (always computed)
        bbox_features[(pid, frame)] = compute_bbox_features(bbox, img_w, img_h)
        labels[(pid, frame)] = int(entry["ttm_label"])
        bboxes_out[(pid, frame)] = np.array(bbox, dtype=np.float32)

        # Gaze features (L2CS-Net)
        if gaze_pipeline is not None:
            try:
                results = gaze_pipeline.step(crop)
                pitch = float(results.pitch[0])
                yaw = float(results.yaw[0])
                gaze_feat = np.array([pitch, yaw, np.sin(pitch), np.cos(pitch),
                                       np.sin(yaw), np.cos(yaw)], dtype=np.float32)
            except Exception:
                gaze_feat = np.zeros(6, dtype=np.float32)
        else:
            gaze_feat = np.zeros(6, dtype=np.float32)
        gaze_features[(pid, frame)] = gaze_feat

    # Extract face features in batch
    if len(crop_images) > 0:
        # Process in sub-batches to avoid OOM
        batch_size = 64
        all_feats = []
        for i in range(0, len(crop_images), batch_size):
            batch = crop_images[i:i + batch_size]
            feats = face_extractor.extract_batch(batch)
            all_feats.append(feats)
        all_feats = np.concatenate(all_feats, axis=0)

        for idx, key in enumerate(crop_keys):
            face_features[key] = all_feats[idx]

    # Extract audio features
    num_frames = max(frame_indices) + 1
    # Get FPS from video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    audio_features = extract_audio_mfcc(video_path, sr=cfg.audio_sr,
                                         n_mfcc=cfg.n_mfcc, fps=fps,
                                         num_frames=num_frames)

    return {
        "face_features": face_features,
        "audio_features": audio_features,
        "bbox_features": bbox_features,
        "gaze_features": gaze_features,
        "labels": labels,
        "bboxes": bboxes_out,
        "metadata": {
            "clip_uid": clip_uid,
            "video_uid": video_uid,
            "person_ids": person_ids,
            "frame_indices": frame_indices,
            "img_w": img_w,
            "img_h": img_h,
            "fps": fps,
        },
    }


def preprocess_clip_lite(clip_uid: str, entries: list, cfg: Config) -> dict | None:
    """
    Lightweight preprocessing using only bbox metadata (no video files needed).
    Good for testing the pipeline and as a baseline.
    """
    video_uid = entries[0]["video_uid"]

    entry_map = {}
    frame_indices = set()
    person_ids = set()
    for e in entries:
        pid = str(e["person_id"])
        frame = int(e["frame"])
        entry_map[(pid, frame)] = e
        frame_indices.add(frame)
        person_ids.add(pid)

    frame_indices = sorted(frame_indices)
    person_ids = sorted(person_ids)

    if len(frame_indices) < cfg.min_frames_per_clip:
        return None

    bbox_features = {}
    gaze_features = {}
    labels = {}
    bboxes_out = {}

    for (pid, frame), entry in entry_map.items():
        bbox = entry["bbox"]
        bbox_features[(pid, frame)] = compute_bbox_features(bbox)
        labels[(pid, frame)] = int(entry["ttm_label"])
        bboxes_out[(pid, frame)] = np.array(bbox, dtype=np.float32)
        gaze_features[(pid, frame)] = np.zeros(6, dtype=np.float32)

    return {
        "face_features": {},       # empty in lite mode
        "audio_features": None,    # empty in lite mode
        "bbox_features": bbox_features,
        "gaze_features": gaze_features,
        "labels": labels,
        "bboxes": bboxes_out,
        "metadata": {
            "clip_uid": clip_uid,
            "video_uid": video_uid,
            "person_ids": person_ids,
            "frame_indices": frame_indices,
            "img_w": 1920,
            "img_h": 1080,
            "fps": 30.0,
        },
    }


def preprocess_split(split: str, cfg: Config):
    """Preprocess an entire data split (train or val)."""
    import time

    json_path = cfg.train_json_path if split == "train" else cfg.val_json_path
    clip_data = load_annotations(json_path)

    save_dir = os.path.join(cfg.feature_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize feature extractor
    face_extractor = None
    gaze_pipeline = None
    if cfg.feature_mode == "full":
        print("Initializing ResNet-50 face feature extractor ...")
        face_extractor = FaceFeatureExtractor(device=cfg.device)
        if HAS_L2CS:
            try:
                gaze_pipeline = GazePipeline(
                    weights=Path(cfg.gaze_model_path),
                    arch='ResNet50',
                    device=torch.device(cfg.device)
                )
                print("  L2CS-Net gaze pipeline initialized")
            except Exception as e:
                print(f"  [WARN] Gaze pipeline init failed: {e}")

    success = 0
    skipped = 0
    already_done = 0
    total = len(clip_data)
    clip_times = []

    split_start = time.time()
    print(f"\n{'='*60}")
    print(f"  Preprocessing {split} split: {total} clips")
    print(f"  Mode: {cfg.feature_mode} | Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for i, (clip_uid, entries) in enumerate(clip_data.items()):
        save_path = os.path.join(save_dir, f"{clip_uid}.pkl")

        # Skip if already processed
        if os.path.isfile(save_path):
            already_done += 1
            success += 1
            continue

        clip_start = time.time()

        try:
            if cfg.feature_mode == "full":
                result = preprocess_clip_full(clip_uid, entries, cfg, face_extractor, gaze_pipeline)
            else:
                result = preprocess_clip_lite(clip_uid, entries, cfg)

            if result is None:
                skipped += 1
                continue

            with open(save_path, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            success += 1

        except Exception as e:
            print(f"  [ERROR] Failed on clip {clip_uid}: {e}")
            skipped += 1

        clip_elapsed = time.time() - clip_start
        clip_times.append(clip_elapsed)

        # Print progress every 10 clips or at milestones
        processed = i + 1 - already_done
        if processed > 0 and (processed % 10 == 0 or processed <= 3 or (i + 1) == total):
            avg_time = sum(clip_times) / len(clip_times)
            remaining = total - (i + 1)
            eta_seconds = remaining * avg_time
            elapsed = time.time() - split_start

            # Format times
            elapsed_str = _format_time(elapsed)
            eta_str = _format_time(eta_seconds)

            print(f"  [{i+1}/{total}] "
                  f"✓{success} ✗{skipped} ⏭{already_done} | "
                  f"Last: {clip_elapsed:.1f}s | Avg: {avg_time:.1f}s/clip | "
                  f"Elapsed: {elapsed_str} | ETA: {eta_str}")

    total_time = time.time() - split_start

    print(f"\n{'='*60}")
    print(f"  {split.upper()} PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed:     {success}/{total} clips")
    print(f"  Already cached: {already_done}")
    print(f"  Skipped:       {skipped}")
    print(f"  Total time:    {_format_time(total_time)}")
    if clip_times:
        print(f"  Avg per clip:  {sum(clip_times)/len(clip_times):.2f}s")
        print(f"  Throughput:    {len(clip_times)/max(total_time,1):.1f} clips/sec")
    print(f"{'='*60}\n")


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Ego4D TTM data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "both"])
    parser.add_argument("--mode", type=str, default="full", choices=["full", "lite"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = Config()
    cfg.feature_mode = args.mode
    cfg.device = args.device
    if args.data_root:
        cfg.data_root = args.data_root
        cfg.annotation_dir = os.path.join(args.data_root, "annotations")
        cfg.clips_dir = os.path.join(args.data_root, "clips")
        cfg.videos_dir = os.path.join(args.data_root, "full_scale")
        cfg.feature_dir = os.path.join(args.data_root, "preprocessed_features")
        os.makedirs(cfg.feature_dir, exist_ok=True)

    if args.split in ("train", "both"):
        preprocess_split("train", cfg)
    if args.split in ("val", "both"):
        preprocess_split("val", cfg)

    print("\nDone! Features saved to:", cfg.feature_dir)
