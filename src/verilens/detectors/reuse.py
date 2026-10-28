"""
Perceptual-Hash Image Reuse Detector & Reference Index.
Computes perceptual visual fingerprints (DCT pHash, dHash, aHash),
indexes reference media corpus, detects out-of-context image reuse,
and extracts EXIF provenance telemetry.
Supports NOT_RUN, UNCERTAIN, ERROR, and SUCCESS statuses.
"""

import io
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Union
from PIL import Image, ExifTags
import numpy as np
import scipy.fftpack

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger
from verilens.core.video import extract_video_frames

logger = get_logger("ImageReuseDetector")


def compute_phash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """
    2D Discrete Cosine Transform (DCT) perceptual hash.
    Resizes image to 32x32 grayscale, computes 2D DCT, extracts top 8x8 low frequencies,
    and thresholds against the median coefficient (excluding DC).
    """
    img_size = hash_size * highfreq_factor
    resized = image.convert("L").resize((img_size, img_size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    dct = scipy.fftpack.dct(scipy.fftpack.dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
    sub = dct[:hash_size, :hash_size]
    # Median of AC coefficients (excluding DC term at (0, 0))
    flat = sub.flatten()
    median_val = float(np.median(flat[1:])) if len(flat) > 1 else float(flat[0])
    bits = "".join(["1" if x > median_val else "0" for x in flat])
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Difference hash based on adjacent horizontal pixel brightness gradients."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = "".join(["1" if b else "0" for b in diff.flatten()])
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def compute_ahash(image: Image.Image, hash_size: int = 8) -> str:
    """Average hash based on pixel intensities relative to the global mean."""
    resized = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(resized, dtype=np.float32)
    avg = float(np.mean(arr))
    bits = "".join(["1" if p > avg else "0" for p in arr.flatten()])
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(hex1: str, hex2: str) -> int:
    """Computes the Hamming bit distance between two hexadecimal perceptual hashes."""
    if not hex1 or not hex2:
        return 64
    try:
        val1 = int(hex1, 16)
        val2 = int(hex2, 16)
        return bin(val1 ^ val2).count("1")
    except ValueError:
        return 64


@dataclass
class ReuseMatch:
    entry_id: str
    hamming_distance: int
    similarity: float
    hash_type: str
    matched_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ImageReuseIndex:
    """
    In-memory perceptual hash index for identifying recycled media
    and cross-referencing provenance archives.
    """

    def __init__(self, entries: Optional[List[Dict[str, Any]]] = None):
        self._entries: Dict[str, Dict[str, Any]] = {}
        if entries:
            for item in entries:
                self.add(**item)

    def add(
        self,
        entry_id: str,
        image: Optional[Union[Image.Image, bytes]] = None,
        phash: Optional[str] = None,
        dhash: Optional[str] = None,
        ahash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Indexes a reference image by its perceptual fingerprints and metadata."""
        if image is not None:
            if isinstance(image, bytes):
                pil_img = Image.open(io.BytesIO(image))
            else:
                pil_img = image
            if phash is None:
                phash = compute_phash(pil_img)
            if dhash is None:
                dhash = compute_dhash(pil_img)
            if ahash is None:
                ahash = compute_ahash(pil_img)

        self._entries[entry_id] = {
            "entry_id": entry_id,
            "phash": phash or "",
            "dhash": dhash or "",
            "ahash": ahash or "",
            "metadata": metadata or {},
        }
        return entry_id

    def search(
        self,
        query_image: Optional[Union[Image.Image, bytes]] = None,
        query_phash: Optional[str] = None,
        query_dhash: Optional[str] = None,
        query_ahash: Optional[str] = None,
        max_distance: int = 10,
        hash_type: str = "phash",
    ) -> List[ReuseMatch]:
        """
        Searches the index for near-duplicate or recycled matches within max_distance bits.
        """
        if query_image is not None:
            if isinstance(query_image, bytes):
                pil_img = Image.open(io.BytesIO(query_image))
            else:
                pil_img = query_image
            if hash_type == "phash" and query_phash is None:
                query_phash = compute_phash(pil_img)
            elif hash_type == "dhash" and query_dhash is None:
                query_dhash = compute_dhash(pil_img)
            elif hash_type == "ahash" and query_ahash is None:
                query_ahash = compute_ahash(pil_img)

        target_hash = (
            query_phash if hash_type == "phash"
            else query_dhash if hash_type == "dhash"
            else query_ahash
        )
        if not target_hash:
            return []

        matches: List[ReuseMatch] = []
        for entry_id, entry in self._entries.items():
            ref_hash = entry.get(hash_type) or ""
            if not ref_hash:
                continue
            dist = hamming_distance(target_hash, ref_hash)
            if dist <= max_distance:
                sim = max(0.0, round((64 - dist) / 64.0, 3))
                matches.append(
                    ReuseMatch(
                        entry_id=entry_id,
                        hamming_distance=dist,
                        similarity=sim,
                        hash_type=hash_type,
                        matched_hash=ref_hash,
                        metadata=entry.get("metadata", {}),
                    )
                )

        matches.sort(key=lambda m: m.hamming_distance)
        return matches

    def count(self) -> int:
        return len(self._entries)

    def clear(self):
        self._entries.clear()

    def list_entries(self) -> List[Dict[str, Any]]:
        return list(self._entries.values())


# Global default index instance pre-seeded with sample historical reference corpus
default_reuse_index = ImageReuseIndex()


class MediaReuseDetector(BaseDetector):
    detector_id = "media_reuse"
    detector_name = "Out-of-Context Media Reuse Detector"
    description = (
        "Generates perceptual hashes (pHash, dHash, aHash), queries reference reuse index, "
        "and examines EXIF provenance to flag recycled or edited media."
    )
    required_modalities = ["image"]

    def __init__(self, index: Optional[ImageReuseIndex] = None):
        self.index: ImageReuseIndex = index or default_reuse_index

    # Static compatibility wrappers
    @staticmethod
    def _compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
        return compute_dhash(image, hash_size)

    @staticmethod
    def _compute_ahash(image: Image.Image, hash_size: int = 8) -> str:
        return compute_ahash(image, hash_size)

    @staticmethod
    def _compute_phash(image: Image.Image, hash_size: int = 8) -> str:
        return compute_phash(image, hash_size)

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        if not input_data.media_bytes:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary="No visual media provided for reuse inspection.",
            ), []

        is_video = input_data.has_video or bool(
            input_data.media_filename and input_data.media_filename.lower().endswith((".mp4", ".mov", ".avi", ".webm", ".mkv"))
        ) or bool(
            input_data.media_content_type and input_data.media_content_type.startswith("video/")
        )

        if is_video:
            try:
                frames = extract_video_frames(input_data.media_bytes, max_frames=1)
                if not frames:
                    return DetectorFinding(
                        detector_id=self.detector_id,
                        detector_name=self.detector_name,
                        status=DetectorStatus.UNCERTAIN,
                        manipulation_score=None,
                        confidence=0.20,
                        summary="Video contains zero decodable frames for reuse indexing.",
                    ), []
                image = Image.fromarray(frames[0])
                width, height = image.size
            except Exception as exc:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Unable to extract video keyframe: {str(exc)}",
                    error_message=str(exc),
                ), []
        else:
            try:
                image = Image.open(io.BytesIO(input_data.media_bytes))
                width, height = image.size
            except Exception as exc:
                return DetectorFinding(
                    detector_id=self.detector_id,
                    detector_name=self.detector_name,
                    status=DetectorStatus.ERROR,
                    manipulation_score=None,
                    confidence=0.0,
                    summary=f"Unable to decode image bytes: {str(exc)}",
                    error_message=str(exc),
                ), []

        # Resolution boundary check (minimum 32x32 for valid provenance hashing)
        if width < 32 or height < 32:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.20,
                summary=f"Image dimensions ({width}x{height}) are too small for reliable provenance hashing.",
                metrics={"width": width, "height": height},
            ), []

        # Degenerate flat image check
        gray_arr = np.asarray(image.convert("L"), dtype=np.float32)
        variance = float(np.var(gray_arr))
        if variance < 1e-4:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.20,
                summary=f"Degenerate flat image with zero variance; perceptual hash is uninformative.",
                metrics={"width": width, "height": height, "variance": 0.0},
            ), []

        phash_val = compute_phash(image)
        dhash_val = compute_dhash(image)
        ahash_val = compute_ahash(image)

        # Inspect EXIF metadata safely
        exif_data: Dict[str, Any] = {}
        editing_software_detected = False
        software_name = ""
        try:
            raw_exif = image.getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, (str, int, float)):
                        exif_data[tag_name] = str(value)
                        if tag_name.lower() in ("software", "processingsoftware"):
                            editing_software_detected = True
                            software_name = str(value)
        except Exception:
            pass  # Non-fatal

        evidence: List[EvidenceItem] = []
        evidence.append(
            EvidenceItem(
                id="ev_perceptual_hash",
                detector_id=self.detector_id,
                evidence_type="PERCEPTUAL_FINGERPRINT",
                description=f"Generated pHash: {phash_val}, dHash: {dhash_val}, aHash: {ahash_val}.",
                severity=EvidenceSeverity.LOW,
                metadata={
                    "phash": phash_val,
                    "dhash": dhash_val,
                    "ahash": ahash_val,
                    "resolution": f"{width}x{height}",
                },
            )
        )

        # Query perceptual hash reuse index
        reuse_matches = self.index.search(query_phash=phash_val, max_distance=10, hash_type="phash")
        match_found = len(reuse_matches) > 0
        best_match = reuse_matches[0] if match_found else None

        manipulation_score = 0.10
        confidence = 0.85
        summary_points = [f"Computed perceptual visual fingerprints ({width}x{height})."]

        if match_found and best_match:
            # Calibrate manipulation score according to visual distance
            if best_match.hamming_distance <= 3:
                # Near-exact match to historical reference
                manipulation_score = 0.85
                confidence = 0.92
                severity = EvidenceSeverity.HIGH
            elif best_match.hamming_distance <= 6:
                manipulation_score = 0.70
                confidence = 0.85
                severity = EvidenceSeverity.MEDIUM
            else:
                manipulation_score = 0.50
                confidence = 0.75
                severity = EvidenceSeverity.LOW

            matched_title = best_match.metadata.get("title", best_match.entry_id)
            matched_date = best_match.metadata.get("original_date", "archived date")
            summary_points.append(
                f"Identified duplicate perceptual match in historical index (ID: {best_match.entry_id}, "
                f"Hamming distance: {best_match.hamming_distance}/64, Similarity: {best_match.similarity * 100:.1f}%)."
            )
            evidence.append(
                EvidenceItem(
                    id="ev_image_reuse",
                    detector_id=self.detector_id,
                    evidence_type="IMAGE_REUSE",
                    description=(
                        f"Image visual fingerprint closely matches archived entry '{matched_title}' "
                        f"({matched_date}) at {best_match.similarity * 100:.1f}% similarity."
                    ),
                    severity=severity,
                    metadata={
                        "matched_entry_id": best_match.entry_id,
                        "hamming_distance": best_match.hamming_distance,
                        "similarity": best_match.similarity,
                        "matched_metadata": best_match.metadata,
                    },
                )
            )

        if editing_software_detected:
            manipulation_score += 0.35
            summary_points.append(f"Detected editor signature: {software_name}.")
            evidence.append(
                EvidenceItem(
                    id="ev_exif_software",
                    detector_id=self.detector_id,
                    evidence_type="METADATA",
                    description=f"EXIF metadata indicates manipulation or export via {software_name}.",
                    severity=EvidenceSeverity.MEDIUM,
                    metadata={"software": software_name},
                )
            )

        if not exif_data:
            summary_points.append("EXIF metadata stripped (common in social media republishing).")

        manipulation_score = round(min(1.0, manipulation_score), 2)

        finding = DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=manipulation_score,
            confidence=confidence,
            summary=" ".join(summary_points),
            metrics={
                "phash": phash_val,
                "dhash": dhash_val,
                "ahash": ahash_val,
                "match_found": match_found,
                "matched_entry_id": best_match.entry_id if best_match else None,
                "hamming_distance": best_match.hamming_distance if best_match else None,
                "similarity": best_match.similarity if best_match else None,
                "exif_tags_found": len(exif_data),
                "has_exif": bool(exif_data),
                "editing_software_detected": editing_software_detected,
                "resolution": f"{width}x{height}",
            },
        )
        return finding, evidence


# Aliases
ImageReuseDetector = MediaReuseDetector
ReuseDetector = MediaReuseDetector
