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

### 2. Configure Environment (Optional Free AI Reasoning & Fact-Checking)
Copy `.env.example` to `.env` to enable Google Gemini 2.5 Flash and Google Fact Check Tools API (both 100% Free):
```bash
cp .env.example .env
```
- **Gemini API Key**: Free at [Google AI Studio](https://aistudio.google.com/) for multimodal reasoning and rich explanations.
- **Fact Check Tools API Key**: Free at [Google Cloud Console](https://console.cloud.google.com/apis/credentials) for live ClaimReview debunks (Reuters, Snopes, AP, USA Today).

*(If left unconfigured, VeriLens AI runs in 100% offline local forensic ensemble mode).*

### 3. Run the Test Suite
```bash
python -m pytest -v
```
*(76 comprehensive automated unit and integration tests passing at 100%).*

### 4. 1-Click Launch (Windows)
Double-click `start.bat` or run:
```powershell
.\start.bat
```
*(To shut down both services: `.\stop.bat`).*

### 5. 1-Command Launch with Docker
```bash
docker compose up --build
```

---

## 🌐 Live Access Points
- **Interactive Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI OpenAPI Swagger Documentation**: `http://localhost:8000/docs`
- **Health Check Endpoint**: `GET http://localhost:8000/health`
- **Multimodal Verification Endpoint**: `POST http://localhost:8000/analyze`

---

## 🎯 Preset Evaluation Scenarios (for Judges)
The Streamlit sidebar includes one-click presets covering key edge cases:
1. **🚨 Synthetic Image + Caption Mismatch**: Demonstrates 2D FFT frequency grid artifact flagging and day/night luminance contradictions.
2. **🔄 Recycled Historical Photo**: Demonstrates perceptual `dHash` generation and EXIF software detection.
3. **✍️ Synthetic AI Text**: Demonstrates statistical burstiness and machine cadence detection.
4. **🛡️ Authentic Natural Media Baseline**: Demonstrates clean execution without false alarms.
5. **⚠️ Low-Resolution Thumbnail & Brief Text**: Demonstrates `UNCERTAIN` status handling when inputs fall below operational thresholds.
6. **🌐 Live Fact-Check Cross-Reference (Shark on Freeway)**: Demonstrates live cross-referencing against Google Fact Check Tools (ClaimReview) with certified debunks from Snopes and USA Today.

---

## 📄 API Contract & Documentation
- **Implementation Plan**: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
- **API Schema Specification**: [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md)
- **Handoff & Architecture Log**: [`HANDOFF.md`](HANDOFF.md)

