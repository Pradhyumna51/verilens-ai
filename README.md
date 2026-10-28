# 🛡️ VeriLens AI

> **Multimodal Misinformation Detection & Human-Understandable Explanation Engine**  
> Built for Hacksploit 2026 (Problem Statement 2: Multi-Modal Misinformation Detection with Explanation Generation)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary
Misinformation increasingly spreads across mixed modalities (e.g. caption-media contradictions, synthetic AI diffusion imagery, out-of-context recycled historical photos, and facial deepfakes). VeriLens AI provides a production-grade multimodal verification pipeline featuring:
- **Strict Status Handling**: Every detector produces explicit `NOT_RUN`, `UNCERTAIN`, `ERROR`, or `SUCCESS` statuses.
- **Stable JSON Response Contract**: Standardized schema across `overall_assessment`, `findings`, `evidence`, `explanation`, and `limitations`.
- **Zero-Hallucination Transparency**: No detector claims model execution or quantitative certainty unless algorithmically validated.
- **Dual Interface**: High-throughput FastAPI REST API gateway + judge-friendly interactive Streamlit dashboard with 1-click evaluation presets.

---

## 🏗️ System Architecture

```
                      +------------------------------------+
                      |     Streamlit Client Dashboard     |
                      |   (Presets, Verdicts, Evidence)    |
                      +-----------------+------------------+
                                        | HTTP / In-Memory
                                        v
                      +------------------------------------+
                      |       FastAPI Backend Server       |
                      |   GET  /health      POST /analyze  |
                      |  (25MB media limit, MIME validation)|
                      +-----------------+------------------+
                                        |
                                        v
                      +------------------------------------+
                      |   Multimodal Orchestration Engine  |
                      |   - Modality Dispatch & Execution  |
                      |   - Evidence Collation & Severity  |
                      |   - Rationale & Limitations Fusion |
                      +-----------------+------------------+
                                        |
      +--------------------+------------+------------+--------------------+
      |                    |                         |                    |
      v                    v                         v                    v
+---------------+  +---------------+       +---------------+    +---------------+
|  Consistency  |  |  Media Reuse  |       |Synthetic Image|    |Synthetic Text |
| Caption vs    |  | Perceptual    |       | 2D FFT & Noise|    | Burstiness &  |
| Image Semantics| | dHash/EXIF    |       | Grid Residuals|    | Style Markers |
+---------------+  +---------------+       +---------------+    +---------------+
                                   \       /
                                    v     v
                             +---------------+
                             |    Deepfake   |
                             | Facial Blends |
                             | & Transitions |
                             +---------------+
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Smoke Test Suite
```bash
python -m pytest tests/test_smoke.py -v
```

### 3. Launch the FastAPI Backend Service
```bash
python -m uvicorn verilens.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation will be accessible at:  
👉 `http://127.0.0.1:8000/docs`

### 4. Launch the Streamlit Evaluation Dashboard
```bash
streamlit run ui/app.py
```
Dashboard will be accessible at:  
👉 `http://localhost:8501`

---

## 🎯 Preset Evaluation Scenarios (for Judges)
The Streamlit sidebar includes one-click presets covering key edge cases:
1. **🚨 Synthetic Image + Caption Mismatch**: Demonstrates 2D FFT frequency grid artifact flagging and day/night luminance contradictions.
2. **🔄 Recycled Historical Photo**: Demonstrates perceptual `dHash` generation and EXIF software detection.
3. **✍️ Synthetic AI Text**: Demonstrates statistical burstiness and machine cadence detection.
4. **🛡️ Authentic Natural Media Baseline**: Demonstrates clean execution without false alarms.
5. **⚠️ Low-Resolution Thumbnail & Brief Text**: Demonstrates `UNCERTAIN` status handling when inputs fall below operational thresholds.

---

## 📄 API Contract & Documentation
- **Implementation Plan**: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- **API Schema Specification**: [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md)
