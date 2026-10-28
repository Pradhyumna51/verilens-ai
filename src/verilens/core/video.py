"""
Shared video processing utilities for VeriLens AI.
Provides bounded, memory-safe frame extraction and transcoding helpers
for multimodal and forensic detectors.
"""

import os
import io
import tempfile
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from verilens.core.logger import get_logger

logger = get_logger("VideoUtils")


def extract_video_frames(
    video_bytes: bytes,
    max_frames: int = 8,
    max_duration_seconds: float = 30.0,
    max_dimension: int = 1080,
) -> List[np.ndarray]:
    """
    Safely extracts a bounded number of evenly spaced RGB frames from video bytes using OpenCV.
    Guarantees resource cleanup via temporary file and VideoCapture release.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV ('opencv-python') is required for video frame extraction.")

    if not video_bytes:
        return []

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        temp_file.write(video_bytes)
        temp_file.close()

        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise ValueError("OpenCV failed to open video container.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if total_frames <= 0:
            raise ValueError("Video contains zero decodable frames.")

        max_allowed_frames = min(total_frames, int(fps * max_duration_seconds))
        sample_count = min(max_frames, max_allowed_frames)
        if sample_count <= 0:
            sample_count = 1

        indices = np.linspace(0, max_allowed_frames - 1, sample_count, dtype=int)
        frames: List[np.ndarray] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb_frame.shape
                if max(h, w) > max_dimension:
                    scale = float(max_dimension) / max(h, w)
                    rgb_frame = cv2.resize(rgb_frame, (int(w * scale), int(h * scale)))
                frames.append(rgb_frame)

        cap.release()
        return frames
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except OSError:
                pass


def frame_to_jpeg_bytes(frame_rgb: np.ndarray, quality: int = 85) -> bytes:
    """Converts an RGB numpy array frame to JPEG bytes."""
    pil_img = Image.fromarray(frame_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
