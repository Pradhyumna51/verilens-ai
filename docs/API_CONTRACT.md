# VeriLens AI: API Contract & Schema Specification
**Version:** `1.0.0`  
**Status:** Stable  
**Protocols:** HTTP/1.1, JSON, Multipart/Form-Data

---

## 1. Overview
The VeriLens AI REST API exposes endpoints for multimodal misinformation analysis across text, images, and video. It produces a structured, evidence-backed evaluation containing:
1. `overall_assessment` (verdict, confidence, risk level, summary)
2. `findings` (individual detector outputs with granular execution statuses)
3. `evidence` (concrete extracted signals: perceptual hashes, frequency anomalies, lexical anomalies)
4. `explanation` (structured human-understandable narrative explaining the classification)
5. `limitations` (transparent boundary disclaimers and untested modalities)

---

## 2. Endpoints

### 2.1 GET `/health`
Health check and system capability inspection.

#### Response: `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2026-10-28T14:30:00Z",
  "limits": {
    "max_media_size_bytes": 26214400,
    "max_media_size_mb": 25.0,
    "max_text_chars": 100000,
    "allowed_media_types": [
      "image/jpeg",
      "image/png",
      "image/webp",
      "video/mp4",
      "video/quicktime"
    ]
  },
  "detectors": [
    {
      "id": "consistency",
      "name": "Cross-Modal Consistency Analyzer",
      "description": "Compares textual claims with visual semantics",
      "required_modalities": ["text", "image"]
    },
    {
      "id": "media_reuse",
      "name": "Out-of-Context Media Reuse Detector",
      "description": "Perceptual hashing, metadata provenance, and duplicate footprint detection",
      "required_modalities": ["image"]
    },
    {
      "id": "synthetic_image",
      "name": "Synthetic Image Detector",
      "description": "Frequency domain artifact analysis (DCT/noise) and AI generation cues",
      "required_modalities": ["image"]
    },
    {
      "id": "synthetic_text",
      "name": "Synthetic Text Detector",
      "description": "Statistical burstiness, perplexity proxy, and AI stylistic patterns",
      "required_modalities": ["text"]
    },
    {
      "id": "deepfake",
      "name": "Deepfake Analysis Engine",
      "description": "Facial boundary blending and temporal artifact analysis",
      "required_modalities": ["image", "video"]
    }
  ]
}
```

---

### 2.2 POST `/analyze`
Analyze multimodal content. Accepts `multipart/form-data` (recommended for file uploads) or `application/json` (for text-only claims).

#### Request Parameters (Multipart Form)
- `text` *(string, optional)*: Caption, headline, or claim text.
- `media_file` *(binary file, optional)*: Image or short video file (up to 25 MB).
- `enabled_detectors` *(comma-separated string, optional)*: E.g., `consistency,media_reuse,synthetic_image,synthetic_text,deepfake`. If omitted, all applicable detectors run.

#### Request Schema (JSON Alternative - Text Only)
```json
{
  "text": "Breaking news: Scientists confirm earth will reverse rotation tomorrow.",
  "enabled_detectors": ["synthetic_text"]
}
```

---

### 2.3 Response Schema (`AnalysisResponse`)

```json
{
  "request_id": "req_8f19da4c",
  "timestamp": "2026-10-28T14:30:05.123Z",
  "processing_time_ms": 142.5,
  "modalities_received": ["text", "image"],
  "overall_assessment": {
    "verdict": "SUSPICIOUS",
    "confidence": 0.82,
    "risk_level": "HIGH",
    "summary": "Content exhibits synthetic visual frequency signatures and cross-modal discrepancies between the text caption and image contents.",
    "primary_concerns": [
      "Image frequency spectrum displays periodic artifacts characteristic of diffusion models.",
      "Text asserts a specific location not evidenced in the visual scene."
    ]
  },
  "findings": {
    "consistency": {
      "detector_id": "consistency",
      "detector_name": "Cross-Modal Consistency Analyzer",
      "status": "SUCCESS",
      "manipulation_score": 0.65,
      "confidence": 0.75,
      "summary": "Moderate semantic divergence detected between caption and visual scene.",
      "metrics": {
        "semantic_similarity": 0.35,
        "entity_overlap_score": 0.20
      },
      "execution_time_ms": 32.1
    },
    "media_reuse": {
      "detector_id": "media_reuse",
      "detector_name": "Out-of-Context Media Reuse Detector",
      "status": "SUCCESS",
      "manipulation_score": 0.15,
      "confidence": 0.85,
      "summary": "Unique perceptual hashes generated. No conflicting historical EXIF metadata found.",
      "metrics": {
        "phash": "bf8c1873e1c3904f",
        "dhash": "336b9e28c465612d",
        "has_exif": true
      },
      "execution_time_ms": 18.4
    },
    "synthetic_image": {
      "detector_id": "synthetic_image",
      "detector_name": "Synthetic Image Detector",
      "status": "SUCCESS",
      "manipulation_score": 0.88,
      "confidence": 0.90,
      "summary": "Frequency spectral analysis reveals high-frequency grid artifacts typical of synthetic generation.",
      "metrics": {
        "high_freq_ratio": 0.74,
        "laplacian_variance": 128.4,
        "grid_artifact_score": 0.89
      },
      "execution_time_ms": 41.2
    },
    "synthetic_text": {
      "detector_id": "synthetic_text",
      "detector_name": "Synthetic Text Detector",
      "status": "UNCERTAIN",
      "manipulation_score": null,
      "confidence": 0.40,
      "summary": "Input text length (8 words) is too short for reliable statistical burstiness or perplexity modeling.",
      "metrics": {
        "word_count": 8,
        "min_recommended_words": 20
      },
      "execution_time_ms": 2.1
    },
    "deepfake": {
      "detector_id": "deepfake",
      "detector_name": "Deepfake Analysis Engine",
      "status": "NOT_RUN",
      "manipulation_score": null,
      "confidence": 0.0,
      "summary": "No facial regions detected in provided media frame; facial boundary analysis skipped.",
      "metrics": {},
      "execution_time_ms": 8.0
    }
  },
  "evidence": [
    {
      "id": "ev_01",
      "detector_id": "synthetic_image",
      "evidence_type": "FREQUENCY_ANOMALY",
      "description": "High-frequency Fourier transform grid anomaly detected at 0.89 certainty.",
      "severity": "HIGH",
      "metadata": {
        "grid_frequency_peak": 48.2
      }
    },
    {
      "id": "ev_02",
      "detector_id": "media_reuse",
      "evidence_type": "PERCEPTUAL_FINGERPRINT",
      "description": "Generated standard perceptual hash: bf8c1873e1c3904f.",
      "severity": "LOW",
      "metadata": {
        "algorithm": "pHash-64"
      }
    }
  ],
  "explanation": {
    "headline": "High probability of synthetic manipulation with cross-modal discrepancy",
    "rationale": "The visual asset presents strong frequency-domain characteristics of modern generative diffusion models. Additionally, the caption makes declarative claims not supported by the visual content.",
    "supporting_points": [
      "Synthetic Image Detector identified abnormal high-frequency spectral clustering (score: 0.88).",
      "Consistency analysis found caption semantic divergence (score: 0.65)."
    ],
    "counter_evidence": [
      "Image does not appear to be a recycled duplicate from known historical archives.",
      "Short caption prevents conclusive statistical text generation determination."
    ]
  },
  "limitations": {
    "disclaimers": [
      "VeriLens AI is a hackathon-grade prototype assisting human evaluators.",
      "Detection metrics are probabilistic and should be corroborated by certified fact-checkers."
    ],
    "untested_modalities": [
      "Deepfake facial landmark analysis was not applicable (no faces detected)."
    ],
    "model_assumptions": [
      "Synthetic image detection relies on spatial-frequency domain properties which can be perturbed by aggressive re-compression."
    ],
    "human_verification_recommended": true
  }
}
```

---

## 3. Data Dictionary

### 3.1 Verdict Values
- `LIKELY_REAL`: Consistent, organic media signals, score < 0.25.
- `UNCERTAIN`: Conflicting or insufficient data signals.
- `SUSPICIOUS`: Notable manipulation cues or mismatches, score between 0.50 and 0.75.
- `LIKELY_MANIPULATED`: Strong, high-confidence synthetic or out-of-context signals, score >= 0.75.

### 3.2 Risk Level Values
- `LOW`: Score < 0.30
- `MEDIUM`: Score 0.30 - 0.60
- `HIGH`: Score 0.60 - 0.85
- `CRITICAL`: Score >= 0.85

### 3.3 Detector Status Values
- `NOT_RUN`: Modality missing or detector disabled.
- `UNCERTAIN`: Input present but below confidence/resolution requirements.
- `ERROR`: Operational failure, malformed format, or runtime issue.
- `SUCCESS`: Complete, verified execution.

---

## 4. HTTP Error Codes & Schemas
- `400 Bad Request`: Neither text nor media was provided in the request.
- `413 Payload Too Large`: Media file exceeds 25 MB or text exceeds 100,000 characters.
- `415 Unsupported Media Type`: File type not in allowed MIME whitelist.
- `422 Unprocessable Content`: Pydantic validation failure.
- `500 Internal Server Error`: Unhandled orchestrator failure.
