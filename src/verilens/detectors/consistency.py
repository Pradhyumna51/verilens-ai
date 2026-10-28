"""
Cross-Modal Consistency Analyzer (Member 2: Multimodal Analysis).
Measures semantic alignment between claims/captions and visual media features.
Integrates lazy-loaded pretrained CLIP for claim-image semantic similarity
and pretrained BLIP for generative image captioning, with robust algorithmic fallback.

NOTE: Per ethical and academic verification guidelines, semantic similarity measures
conceptual correspondence between text and imagery—it does NOT prove the real-world
claim is factually true or false.
"""

import io
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from verilens.detectors.base import BaseDetector, DetectorInput
from verilens.schemas.response import (
    DetectorStatus,
    DetectorFinding,
    EvidenceItem,
    EvidenceSeverity,
)
from verilens.core.logger import get_logger

logger = get_logger("ConsistencyDetector")

class ConsistencyDetector(BaseDetector):
    """
    Cross-Modal Consistency Analyzer.
    Evaluates semantic correlation between textual claims and visual content.
    Leverages pretrained CLIP for embedding distance and BLIP for conditional
    image caption generation, falling back to scene-chromatic analysis when
    neural weights are offline.
    """
    detector_id = "consistency"
    detector_name = "Cross-Modal Consistency Analyzer"
    description = (
        "Compares claim concepts against visual semantics using pretrained CLIP embeddings "
        "and BLIP image captioning, augmented by environmental scene verification."
    )
    required_modalities = ["text", "image"]

    def __init__(self):
        # CLIP lazy-load cache
        self._clip_model = None
        self._clip_processor = None
        self._clip_attempted = False

        # BLIP lazy-load cache
        self._captioner_model = None
        self._captioner_processor = None
        self._captioner_attempted = False

    def _get_lazy_clip(self):
        """Lazy loader for pretrained CLIP model and processor (LAION CLIP / OpenAI CLIP)."""
        if not self._clip_attempted:
            self._clip_attempted = True
            try:
                from transformers import CLIPProcessor, CLIPModel
                model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                logger.info(f"Attempting to load CLIP model: {model_name}")
                self._clip_processor = CLIPProcessor.from_pretrained(model_name)
                self._clip_model = CLIPModel.from_pretrained(model_name)
                self._clip_model.eval()
                logger.info("Successfully loaded CLIP model.")
            except Exception as e:
                logger.warning(f"CLIP integration unavailable, using algorithmic fallback: {e}")
                self._clip_model = None
                self._clip_processor = None
        return self._clip_model, self._clip_processor

    def _get_lazy_captioner(self):
        """Lazy loader for pretrained BLIP image captioning model."""
        if not self._captioner_attempted:
            self._captioner_attempted = True
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                model_name = "Salesforce/blip-image-captioning-base"
                logger.info(f"Attempting to load BLIP captioner: {model_name}")
                self._captioner_processor = BlipProcessor.from_pretrained(model_name)
                self._captioner_model = BlipForConditionalGeneration.from_pretrained(model_name)
                self._captioner_model.eval()
                logger.info("Successfully loaded BLIP captioning model.")
            except Exception as e:
                logger.warning(f"BLIP captioner unavailable, using scene descriptor fallback: {e}")
                self._captioner_model = None
                self._captioner_processor = None
        return self._captioner_model, self._captioner_processor

    def _execute(self, input_data: DetectorInput) -> Tuple[DetectorFinding, List[EvidenceItem]]:
        text = (input_data.text or "").strip()
        text_lower = text.lower()

        if not input_data.media_bytes:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.NOT_RUN,
                manipulation_score=None,
                confidence=0.0,
                summary="Skipped: Both text and image modalities are required for cross-modal consistency analysis.",
            ), []

        # Check for non-descriptive or insufficient captions
        tokens = re.findall(r"\b\w+\b", text_lower)
        if len(tokens) < 3 or text_lower in {"image", "photo", "look at this", "see this", "check this out"}:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.30,
                summary="Caption contains insufficient descriptive entities to establish cross-modal correspondence.",
                metrics={
                    "token_count": len(tokens),
                    "epistemic_disclaimer": "Semantic alignment requires at least 3 descriptive tokens.",
                },
            ), []

        # Validate image decoding
        try:
            image = Image.open(io.BytesIO(input_data.media_bytes)).convert("RGB")
            img_arr = np.array(image, dtype=np.float32)
        except Exception as exc:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.ERROR,
                manipulation_score=None,
                confidence=0.0,
                summary=f"Visual decoding error during consistency check: {str(exc)}",
                error_message=str(exc),
            ), []

        # Check for zero-variance flat color images
        img_variance = float(np.var(img_arr))
        if img_variance < 1e-4:
            return DetectorFinding(
                detector_id=self.detector_id,
                detector_name=self.detector_name,
                status=DetectorStatus.UNCERTAIN,
                manipulation_score=None,
                confidence=0.20,
                summary="Image exhibits zero textural variance; cannot compute meaningful semantic correspondence.",
                metrics={"image_variance": img_variance},
            ), []

        evidence: List[EvidenceItem] = []
        model_observations: List[str] = []
        metrics: Dict[str, Any] = {
            "epistemic_disclaimer": (
                "Semantic similarity measures conceptual and visual correspondence; "
                "it does NOT prove whether the factual claim is true or false."
            )
        }

        # -------------------------------------------------------------
        # 1. Pretrained CLIP Semantic Similarity
        # -------------------------------------------------------------
        clip_model, clip_processor = self._get_lazy_clip()
        clip_similarity: Optional[float] = None
        clip_backend = "algorithmic_fallback"

        if clip_model and clip_processor:
            try:
                import torch
                inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                    txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                    raw_sim = float(torch.matmul(txt_emb, img_emb.t())[0][0].item())
                    # Rescale [-1, 1] cosine range to normalized [0, 1]
                    clip_similarity = max(0.0, min(1.0, (raw_sim + 0.2) / 1.2))
                    clip_backend = "pretrained_clip"
            except Exception as e:
                logger.warning(f"Error during CLIP inference: {e}. Falling back to algorithmic analysis.")
                clip_similarity = None

        if clip_similarity is None:
            clip_similarity = self._compute_algorithmic_clip_proxy(text_lower, img_arr)

        metrics["clip_similarity"] = round(clip_similarity, 3)
        metrics["clip_backend"] = clip_backend

        # -------------------------------------------------------------
        # 2. Pretrained BLIP Image Captioning & Claim Alignment
        # -------------------------------------------------------------
        blip_model, blip_processor = self._get_lazy_captioner()
        generated_caption: str = ""
        caption_backend = "scene_descriptor_fallback"

        if blip_model and blip_processor:
            try:
                inputs = blip_processor(image, return_tensors="pt")
                out = blip_model.generate(**inputs, max_new_tokens=40)
                generated_caption = blip_processor.decode(out[0], skip_special_tokens=True).strip()
                caption_backend = "pretrained_blip"
            except Exception as e:
                logger.warning(f"BLIP generation failed: {e}. Using scene descriptor.")
                generated_caption = self._synthesize_scene_descriptor(img_arr)
        else:
            generated_caption = self._synthesize_scene_descriptor(img_arr)

        metrics["generated_caption"] = generated_caption
        metrics["caption_backend"] = caption_backend

        # Calculate word overlap between generated caption and input claim
        caption_words = set(re.findall(r"\b\w{3,}\b", generated_caption.lower()))
        stopwords = {"the", "and", "with", "for", "that", "this", "from", "image", "photo", "scene"}
        caption_keywords = caption_words - stopwords
        claim_keywords = set(re.findall(r"\b\w{3,}\b", text_lower)) - stopwords

        overlap_ratio = 1.0
        if caption_keywords and claim_keywords:
            intersection = caption_keywords.intersection(claim_keywords)
            overlap_ratio = len(intersection) / max(1, min(len(caption_keywords), len(claim_keywords)))
        metrics["caption_claim_overlap"] = round(overlap_ratio, 3)

        # -------------------------------------------------------------
        # 3. Environmental & Chromatic Telemetry (Original Baseline)
        # -------------------------------------------------------------
        mean_r = float(np.mean(img_arr[:, :, 0]))
        mean_g = float(np.mean(img_arr[:, :, 1]))
        mean_b = float(np.mean(img_arr[:, :, 2]))
        luminance = float(0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b)
        metrics["mean_luminance"] = round(luminance, 2)
        metrics["rgb_means"] = [round(mean_r, 1), round(mean_g, 1), round(mean_b, 1)]

        night_keywords = {"night", "dark", "midnight", "dusk", "evening", "moonlit", "stargazing"}
        day_keywords = {"daylight", "sunny", "noon", "bright sun", "clear sky", "morning sun"}
        fire_keywords = {"fire", "wildfire", "flames", "burning", "explosion", "inferno"}
        water_keywords = {"ocean", "sea", "flood", "submerged", "river", "lake", "drowning"}

        text_has_night = any(k in text_lower for k in night_keywords)
        text_has_day = any(k in text_lower for k in day_keywords)
        text_has_fire = any(k in text_lower for k in fire_keywords)
        text_has_water = any(k in text_lower for k in water_keywords)

        mismatches: List[str] = []
        mismatch_penalty = 0.0

        # Luminance contradiction
        if text_has_night and luminance > 160.0:
            mismatches.append(f"Caption claims nighttime/dark setting, but visual media exhibits high daytime luminance ({luminance:.1f}).")
            mismatch_penalty += 0.45
            evidence.append(
                EvidenceItem(
                    id="ev_mismatch_luminance",
                    detector_id=self.detector_id,
                    evidence_type="CROSS_MODAL_MISMATCH",
                    description=f"Inversion between night assertion and daylight luminance ({luminance:.1f}).",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"observed_luminance": round(luminance, 1), "claimed_environment": "night"},
                )
            )
        elif text_has_day and luminance < 40.0:
            mismatches.append(f"Caption claims sunny/daylight conditions, but visual scene has low luminance ({luminance:.1f}).")
            mismatch_penalty += 0.40
            evidence.append(
                EvidenceItem(
                    id="ev_mismatch_darkness",
                    detector_id=self.detector_id,
                    evidence_type="CROSS_MODAL_MISMATCH",
                    description=f"Daylight claim contradicts low visual luminance ({luminance:.1f}).",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"observed_luminance": round(luminance, 1), "claimed_environment": "daylight"},
                )
            )

        # Fire / chromatic warmth contradiction
        if text_has_fire:
            red_ratio = mean_r / (mean_b + 1e-5)
            if red_ratio < 0.95 and mean_b > 110.0:
                mismatches.append(f"Caption asserts active fire/inferno, but visual spectrum is predominantly blue/cool (R/B ratio: {red_ratio:.2f}).")
                mismatch_penalty += 0.40
                evidence.append(
                    EvidenceItem(
                        id="ev_mismatch_fire_spectrum",
                        detector_id=self.detector_id,
                        evidence_type="CROSS_MODAL_MISMATCH",
                        description=f"Absence of warm chromatic signature for claimed fire event (R/B ratio: {red_ratio:.2f}).",
                        severity=EvidenceSeverity.HIGH,
                        metadata={"rb_ratio": round(red_ratio, 2)},
                    )
                )

        # -------------------------------------------------------------
        # 4. Synthesizing Scores & Observations
        # -------------------------------------------------------------
        # CLIP evidence
        if clip_similarity < 0.30:
            evidence.append(
                EvidenceItem(
                    id="ev_clip_semantic_divergence",
                    detector_id=self.detector_id,
                    evidence_type="CLIP_SEMANTIC_DIVERGENCE",
                    description=f"Low semantic similarity ({clip_similarity:.2f}) between text concepts and visual content.",
                    severity=EvidenceSeverity.HIGH,
                    metadata={"clip_similarity": round(clip_similarity, 3), "backend": clip_backend},
                )
            )
            model_observations.append(f"CLIP embedding divergence observed (similarity: {clip_similarity:.2f}).")
            mismatch_penalty += 0.35
        elif clip_similarity >= 0.70:
            evidence.append(
                EvidenceItem(
                    id="ev_clip_semantic_alignment",
                    detector_id=self.detector_id,
                    evidence_type="CLIP_SEMANTIC_ALIGNMENT",
                    description=f"High conceptual alignment ({clip_similarity:.2f}) between text claim and visual features.",
                    severity=EvidenceSeverity.LOW,
                    metadata={"clip_similarity": round(clip_similarity, 3), "backend": clip_backend},
                )
            )
            model_observations.append(f"CLIP embeddings exhibit strong semantic correspondence ({clip_similarity:.2f}).")

        # BLIP caption evidence
        if generated_caption:
            if overlap_ratio < 0.15 and len(claim_keywords) >= 4:
                evidence.append(
                    EvidenceItem(
                        id="ev_blip_caption_discrepancy",
                        detector_id=self.detector_id,
                        evidence_type="BLIP_CAPTION_DISCREPANCY",
                        description=f"Generated visual caption ('{generated_caption}') diverges from provided claim keywords.",
                        severity=EvidenceSeverity.MEDIUM,
                        metadata={"generated_caption": generated_caption, "overlap_ratio": round(overlap_ratio, 2)},
                    )
                )
                model_observations.append(f"Generated visual description ('{generated_caption}') shows low keyword overlap with claim.")
                mismatch_penalty += 0.20
            else:
                evidence.append(
                    EvidenceItem(
                        id="ev_blip_caption_recorded",
                        detector_id=self.detector_id,
                        evidence_type="BLIP_CAPTION_OBSERVATION",
                        description=f"Generated visual description: '{generated_caption}'.",
                        severity=EvidenceSeverity.LOW,
                        metadata={"generated_caption": generated_caption},
                    )
                )

        # Compute manipulation score (higher = greater discrepancy/manipulation probability)
        # Note: manipulation score measures out-of-context or visual-claim inconsistency,
        # not the absolute factual truth of the underlying real-world event.
        clip_discrepancy = 1.0 - clip_similarity
        combined_score = max(clip_discrepancy * 0.5 + mismatch_penalty * 0.5, mismatch_penalty)
        manipulation_score = float(np.clip(combined_score, 0.10, 0.95))

        confidence = 0.84 if (clip_backend == "pretrained_clip" or caption_backend == "pretrained_blip") else 0.76

        if mismatches:
            summary = (
                f"Cross-modal discrepancies detected: {'; '.join(mismatches[:2])}. "
                f"CLIP similarity: {clip_similarity:.2f}. "
                f"(Note: Similarity reflects text-visual conceptual alignment, not ground-truth veracity)."
            )
        elif clip_similarity < 0.30:
            summary = (
                f"Semantic divergence detected between claim and visual media (CLIP similarity: {clip_similarity:.2f}). "
                f"Generated description: '{generated_caption}'."
            )
        else:
            summary = (
                f"Cross-modal consistency validated (CLIP similarity: {clip_similarity:.2f}, luminance: {luminance:.1f}). "
                f"Generated description: '{generated_caption}'."
            )

        metrics["mismatches"] = mismatches
        metrics["model_observations"] = model_observations

        return DetectorFinding(
            detector_id=self.detector_id,
            detector_name=self.detector_name,
            status=DetectorStatus.SUCCESS,
            manipulation_score=round(manipulation_score, 2),
            confidence=round(confidence, 2),
            summary=summary,
            metrics=metrics,
        ), evidence

    def _compute_algorithmic_clip_proxy(self, text_lower: str, img_arr: np.ndarray) -> float:
        """Fallback proxy for semantic alignment based on scene-chromatic features."""
        mean_r = float(np.mean(img_arr[:, :, 0]))
        mean_g = float(np.mean(img_arr[:, :, 1]))
        mean_b = float(np.mean(img_arr[:, :, 2]))
        luminance = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

        score = 0.65

        # Night vs Daylight
        if any(w in text_lower for w in ["night", "midnight", "dark", "moon"]):
            if luminance > 150:
                score -= 0.35
            elif luminance < 60:
                score += 0.20

        # Fire / Heat
        if any(w in text_lower for w in ["fire", "blaze", "flames", "burning", "explosion"]):
            red_ratio = mean_r / (mean_b + 1e-5)
            if red_ratio > 1.2:
                score += 0.20
            elif red_ratio < 0.8:
                score -= 0.35

        # Water / Aquatic
        if any(w in text_lower for w in ["ocean", "sea", "lake", "water", "river", "flood"]):
            blue_ratio = mean_b / (mean_r + 1e-5)
            if blue_ratio > 1.05:
                score += 0.15
            elif blue_ratio < 0.8:
                score -= 0.25

        # Forest / Nature
        if any(w in text_lower for w in ["forest", "nature", "trees", "grass", "park"]):
            green_ratio = mean_g / (mean_r + 1e-5)
            if green_ratio > 1.05:
                score += 0.15

        return round(float(np.clip(score, 0.08, 0.92)), 2)

    def _synthesize_scene_descriptor(self, img_arr: np.ndarray) -> str:
        """Generates an interpretable scene observation from image statistics."""
        mean_r = float(np.mean(img_arr[:, :, 0]))
        mean_g = float(np.mean(img_arr[:, :, 1]))
        mean_b = float(np.mean(img_arr[:, :, 2]))
        luminance = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

        time_cue = "nighttime" if luminance < 60 else "daylight" if luminance > 140 else "ambient light"
        if mean_b > 1.15 * max(mean_r, mean_g):
            palette = "predominantly cool blue"
        elif mean_r > 1.25 * max(mean_g, mean_b):
            palette = "warm reddish"
        elif mean_g > 1.15 * max(mean_r, mean_b):
            palette = "green vegetation"
        else:
            palette = "balanced"

        return f"A visual scene captured in {time_cue} with {palette} composition."
