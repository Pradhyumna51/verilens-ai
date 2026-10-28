# VeriLens AI: Implementation Plan
**24-Hour Hackathon Architecture: Multimodal Misinformation Detection & Explanation Engine**

## 1. Executive Summary & Problem Context
Misinformation in 2026 predominantly travels across modalities (caption-image mismatch, AI-generated synthetic media, recycled historical photos paired with breaking news headlines, and deepfake manipulation). 
VeriLens AI delivers a minimal, robust, modular pipeline capable of:
1. Handling multimodal inputs (text claims/captions, images, video clips).
2. Coordinating independent, specialized detectors via a typed registry.
3. Supporting explicit operational states per detector: `NOT_RUN`, `UNCERTAIN`, `ERROR`, and `SUCCESS`.
4. Synthesizing disparate signals into a unified, human-interpretable assessment with concrete evidence citations, natural language rationale, and clear operational limitations.
5. Providing both a developer-grade REST API (FastAPI) and an intuitive, judge-friendly dashboard (Streamlit).

---

## 2. System Architecture

```
                    +------------------------------------+
                    |  Streamlit UI / Client Application |
                    +-----------------+------------------+
                                      | HTTP / Multipart
                                      v
                    +------------------------------------+
                    |        FastAPI Gateway Server      |
                    |   /health              /analyze    |
                    | (Upload Limits, Validation, CORS)  |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |   Multimodal Orchestrator Engine   |
                    |   - Preprocessing & Input Parsing  |
                    |   - Dynamic Detector Dispatch      |
                    |   - Signal Fusion & Assessment     |
                    |   - Evidence & Rationale Synthesizer|
                    +-----------------+------------------+
                                      |
            +-------------------------+-------------------------+
            |                         |                         |
            v                         v                         v
+-----------------------+ +-----------------------+ +-----------------------+
|  ConsistencyDetector  | |   MediaReuseDetector  | | SyntheticImageDetector|
| (Caption vs Media     | | (pHash, dHash, EXIF,  | | (Frequency DCT, noise |
|  Semantic Alignment)  | |  Reverse Provenance)  | |  residuals, ViT hook) |
+-----------------------+ +-----------------------+ +-----------------------+
            |                         |
            v                         v
+-----------------------+ +-----------------------+
| SyntheticTextDetector | |   DeepfakeDetector    |
| (Perplexity proxy,    | | (Facial artifact cues,|
|  burstiness, markers) | |  temporal boundaries) |
+-----------------------+ +-----------------------+
```

---

## 3. Core Modules & Directory Layout

```
verilens-ai/
├── IMPLEMENTATION_PLAN.md          # 24-hr roadmap, design decisions, verification criteria
├── docs/
│   └── API_CONTRACT.md             # Canonical OpenAPI / JSON payload specification
├── requirements.txt                # Production and development dependencies
├── src/
│   └── verilens/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py           # Upload limits (e.g., 25MB media, 100KB text), thresholds
│       │   └── logger.py           # Structured logging
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── response.py         # Pydantic models: AnalysisResponse, OverallAssessment, Finding, EvidenceItem
│       ├── detectors/
│       │   ├── __init__.py
│       │   ├── base.py             # BaseDetector abstract interface, DetectorStatus enum
│       │   ├── consistency.py      # Caption vs visual semantic consistency
│       │   ├── media_reuse.py      # Perceptual hashes, EXIF provenance, duplicate detection
│       │   ├── synthetic_image.py  # Frequency analysis (DCT/Laplacian) + lazy pretrained classifier hook
│       │   ├── synthetic_text.py   # Statistical burstiness/perplexity + stylistic markers
│       │   ├── deepfake.py         # Face boundary artifact cues & landmark inconsistency
│       │   └── registry.py         # Detector registration, lifecycle, and parallel execution
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── orchestrator.py     # Multimodal fusion, confidence scoring, evidence collation
│       └── api/
│           ├── __init__.py
│           └── app.py              # FastAPI endpoints (/health, /analyze)
├── ui/
│   ├── __init__.py
│   └── app.py                      # Interactive Streamlit dashboard with preset test cases
└── tests/
    ├── __init__.py
    └── test_smoke.py               # Comprehensive pytest test suite validating endpoints, statuses & schemas
```

---

## 4. Detector Status Specifications

To maintain academic rigor and prevent false claims during evaluation, every detector must emit one of four distinct states:

| Status | Trigger Condition | Semantics |
|---|---|---|
| `NOT_RUN` | Required modality omitted by user (e.g., no image provided to `SyntheticImageDetector`) or detector explicitly disabled. | Omitted from negative evidence calculation. |
| `UNCERTAIN` | Modality provided, but input is ambiguous, degraded, below operational threshold (e.g. text < 15 words, low-res thumbnail, no faces detected). | Flagged to user; does not skew score falsely toward 0.0 or 1.0. |
| `ERROR` | Malformed media format, processing timeout, or missing optional runtime dependency. | Gracefully caught and logged; includes diagnostic message without crashing pipeline. |
| `SUCCESS` | Detector executed algorithmically; produced quantitative metric, confidence interval, and extractable evidence. | Active contributor to final verdict and rationale. |

---

## 5. Phased 24-Hour Execution Roadmap

### Phase 1: Contract, Architecture & Skeleton (Hours 0 - 4) — *CURRENT PHASE*
- Establish canonical schema contract (`docs/API_CONTRACT.md`) and implementation blueprint.
- Implement typed Pydantic models with validation.
- Implement detector base interfaces, status enums, and registry.
- Build resilient FastAPI server (`/health`, `/analyze`) with multipart upload handling, payload sanitization, and size limits (25MB media, 100KB text).
- Build Streamlit UI working with direct engine and API modes.
- Implement comprehensive smoke test suite and verify test execution.

### Phase 2: Lightweight Algorithmic Implementations (Hours 4 - 10)
- **Synthetic Text**: Implement statistical metrics (type-token ratio, sentence length variance, repetition penalty, burstiness index) without downloading heavy 5GB LLMs.
- **Media Reuse**: Implement perceptual hashing (`pHash`, `dHash`, average hash) using Pillow, plus standard EXIF metadata extraction and provenance timestamp verification.
- **Synthetic Image**: Implement Discrete Cosine Transform (DCT) frequency grid analysis and noise residual Laplacian variance to detect diffusion artifact fingerprints.
- **Consistency**: Implement cross-modal lexical overlap and color/scene keyword matching; provide lazy-loaded CLIP hook for semantic embeddings.
- **Deepfake**: Implement facial region-of-interest analysis with lazy-loaded landmark cues.

### Phase 3: Pretrained Integrations & Robustness (Hours 10 - 18)
- Wire optional Hugging Face transformers with lazy imports so the app starts instantly without blocking on multi-gigabyte model weights.
- Implement adversarial perturbation checks (noise injection, JPEG compression tolerance).
- Add benchmark evaluation script for precision/recall across sample test sets.

### Phase 4: UI Polish, Presets & Demo Packaging (Hours 18 - 24)
- Add one-click test presets into Streamlit (Synthetic Image, Breaking News Recycle, Out-of-Context Quote, Authentic Baseline).
- Refine evidence visualization cards, radar charts, and exportable JSON report.
- Deliver final pitch walkthrough and recording.

---

## 6. Verification & Quality Gates
- **Smoke Tests**: Verify `/health` and `/analyze` across all four detector statuses (`NOT_RUN`, `UNCERTAIN`, `ERROR`, `SUCCESS`).
- **Error Handling**: Verify corrupted files and oversized inputs yield standard HTTP 400/413/422 responses with structured error bodies.
- **Zero Hallucination Rule**: No detector reports `SUCCESS` or claims model certainty unless it has executed its underlying verification pass.
