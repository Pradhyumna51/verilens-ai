"""
VeriLens AI - Streamlit Dashboard
Multimodal Misinformation Analysis & Explanation Engine
"""

import io
import json
import base64
import sys
from pathlib import Path
import requests
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

# Ensure src/ is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verilens.core.config import settings
from verilens.detectors.registry import registry
from verilens.pipeline.orchestrator import pipeline
from verilens.schemas.response import AnalysisResponse, DetectorStatus, Verdict, RiskLevel

# Page configuration
st.set_page_config(
    page_title="VeriLens AI | Multimodal Misinformation Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark tech aesthetics, glassmorphism, glowing badges)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(18, 22, 34, 0.98), rgba(10, 12, 18, 1.0));
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    
    .verdict-card {
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .verdict-LIKELY_MANIPULATED {
        background: linear-gradient(135deg, rgba(225, 29, 72, 0.15) 0%, rgba(136, 19, 55, 0.25) 100%);
        border-left: 6px solid #f43f5e;
    }
    
    .verdict-SUSPICIOUS {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.25) 100%);
        border-left: 6px solid #fbbf24;
    }
    
    .verdict-LIKELY_REAL {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.25) 100%);
        border-left: 6px solid #10b981;
    }
    
    .verdict-UNCERTAIN {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(76, 29, 149, 0.25) 100%);
        border-left: 6px solid #a78bfa;
    }
    
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .badge-status-SUCCESS { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-status-UNCERTAIN { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-status-ERROR { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-status-NOT_RUN { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid #64748b; }

    .detector-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-pill-online { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #10b981; }
    .status-pill-offline { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid #f59e0b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper function to generate test images for quick presets
def generate_sample_image(sample_type: str) -> bytes:
    buf = io.BytesIO()
    if sample_type == "synthetic_grid":
        # Create an image with high-frequency periodic pattern
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                val = int((np.sin(x / 2.0) * np.cos(y / 2.0) + 1.0) * 127)
                arr[y, x] = [int(val * 0.7), (val * 2) % 255, 255 - int(val * 0.3)]
        img = Image.fromarray(arr)
        img.save(buf, format="PNG")
    elif sample_type == "recycled_news":
        # Create standard landscape image with EXIF mock software
        img = Image.new("RGB", (320, 240), color=(40, 90, 160))
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 280, 200], fill=(70, 130, 200), outline=(255, 255, 255))
        draw.text((60, 110), "HISTORICAL ARCHIVE 2019", fill=(255, 255, 255))
        exif = img.getexif()
        exif[0x0131] = "Historical Archive Suite 2019"
        img.save(buf, format="JPEG", exif=exif)
    elif sample_type == "natural_photo":
        # Clean photographic landscape with daylight sky and green hill
        img = Image.new("RGB", (256, 256))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 256, 128], fill=(135, 206, 235))
        draw.rectangle([0, 128, 256, 256], fill=(34, 139, 34))
        img.save(buf, format="JPEG", quality=92)
    elif sample_type == "tiny_thumbnail":
        # Under threshold resolution
        img = Image.new("RGB", (20, 20), color=(128, 128, 128))
        img.save(buf, format="PNG")
    return buf.getvalue()

def check_api_health(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=1.2)
        return r.status_code == 200
    except Exception:
        return False

# Header
st.markdown(
    """
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    🛡️ VeriLens AI
                </h1>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 1.05rem;">
                    Multimodal Misinformation Detection & Explanation Engine &bull; Hacksploit 2026
                </p>
            </div>
            <div style="text-align: right;">
                <span class="badge badge-status-SUCCESS">v1.0.0 Stable</span>
                <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #64748b;">FastAPI + Streamlit</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Engine Configuration")
    execution_mode = st.radio(
        "Execution Mode",
        ["Direct Orchestrator (In-Memory)", "FastAPI REST API (http://127.0.0.1:8000)"],
        index=0,
    )
    api_url = st.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")

    # Backend connectivity check
    is_api_online = check_api_health(api_url)
    if is_api_online:
        st.markdown(
            f'<div class="status-pill status-pill-online">● Backend Online ({api_url})</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-pill-offline">● Backend Offline &mdash; Mock Fallback Ready</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🎯 Quick Evaluation Presets")
    st.caption("One-click scenarios designed for hackathon judges:")
    
    preset = st.selectbox(
        "Load Preset Case",
        [
            "Custom Input",
            "🚨 1: Synthetic Image + Misleading Caption",
            "🔄 2: Recycled Historical Photo (Media Reuse)",
            "✍️ 3: Synthetic AI Text (Monotone Essay)",
            "🛡️ 4: Authentic Natural Media Baseline",
            "⚠️ 5: Low-Res Thumbnail (UNCERTAIN status)",
        ],
    )

    st.markdown("---")
    st.markdown("### 🧩 Active Detectors")
    available_detectors = registry.list_all()
    enabled_detector_ids = []
    for d in available_detectors:
        default_checked = True
        checked = st.checkbox(f"{d.detector_name}", value=default_checked, key=f"det_{d.detector_id}")
        if checked:
            enabled_detector_ids.append(d.detector_id)

    st.markdown("---")
    st.caption(f"Upload Limit: {settings.MAX_MEDIA_SIZE_BYTES // (1024*1024)} MB | Max Text: {settings.MAX_TEXT_CHARS:,} chars")

# Preset parameter population
preset_text = ""
preset_image_bytes = None
preset_image_name = None

if preset == "🚨 1: Synthetic Image + Misleading Caption":
    preset_text = "Breaking News: Dangerous firestorm engulfs the downtown metropolitan subway station during midnight hours."
    preset_image_bytes = generate_sample_image("synthetic_grid")
    preset_image_name = "synthetic_diffusion_sample.png"
elif preset == "🔄 2: Recycled Historical Photo (Media Reuse)":
    preset_text = "Urgent: Current ongoing coastal disaster reported just 10 minutes ago."
    preset_image_bytes = generate_sample_image("recycled_news")
    preset_image_name = "recycled_archive_2019.jpg"
elif preset == "✍️ 3: Synthetic AI Text (Monotone Essay)":
    preset_text = (
        "In conclusion, it is important to remember that modern technological frameworks delve into a rich tapestry of innovation. "
        "Furthermore, artificial intelligence plays a crucial role across interconnected systems. "
        "Moreover, digital infrastructure serves as a profound testament to ongoing scientific evolution. "
        "Therefore, one must appreciate the multifaceted dynamics governing our contemporary algorithmic reality."
    )
elif preset == "🛡️ 4: Authentic Natural Media Baseline":
    preset_text = "A serene view of an afternoon lake scene captured during a weekend hike across the national park."
    preset_image_bytes = generate_sample_image("natural_photo")
    preset_image_name = "natural_landscape.jpg"
elif preset == "⚠️ 5: Low-Res Thumbnail (UNCERTAIN status)":
    preset_text = "Short text."
    preset_image_bytes = generate_sample_image("tiny_thumbnail")
    preset_image_name = "pixel_thumbnail.png"

# Session state handling for preset text sync
if "last_selected_preset" not in st.session_state:
    st.session_state["last_selected_preset"] = preset
    st.session_state["active_input_text"] = preset_text

if preset != st.session_state["last_selected_preset"]:
    st.session_state["last_selected_preset"] = preset
    st.session_state["active_input_text"] = preset_text

# Main Layout
col_input, col_results = st.columns([1, 1.25], gap="large")

with col_input:
    st.subheader("📥 Multimodal Input")
    
    input_text = st.text_area(
        "Headline / Claim / Caption Text",
        value=st.session_state.get("active_input_text", preset_text),
        height=140,
        placeholder="Enter news text, social media caption, or claim to analyze...",
    )
    
    uploaded_file = st.file_uploader(
        "Upload Media Asset (Image or Video)",
        type=["png", "jpg", "jpeg", "webp", "mp4", "mov"],
        help=f"Supported: Images and short video clips up to {settings.MAX_MEDIA_SIZE_BYTES // (1024*1024)}MB",
    )

    # Resolve active media
    active_media_bytes = None
    active_media_name = None
    active_media_type = None

    if uploaded_file is not None:
        active_media_bytes = uploaded_file.read()
        active_media_name = uploaded_file.name
        active_media_type = uploaded_file.type
    elif preset_image_bytes is not None:
        active_media_bytes = preset_image_bytes
        active_media_name = preset_image_name
        active_media_type = "image/png" if preset_image_name.endswith(".png") else "image/jpeg"

    if active_media_bytes and active_media_type and active_media_type.startswith("image/"):
        st.image(active_media_bytes, caption=f"Loaded Media: {active_media_name}", use_container_width=True)
    elif active_media_bytes and active_media_type and active_media_type.startswith("video/"):
        st.video(active_media_bytes)

    analyze_btn = st.button("🚀 Run Multimodal Verification", type="primary", use_container_width=True)

with col_results:
    st.subheader("📊 Verification Report")
    
    if analyze_btn:
        if not input_text and not active_media_bytes:
            st.warning("Please supply at least one input modality (text or media file).")
        else:
            with st.spinner("Executing active detector ensemble..."):
                response_data = None
                fallback_engaged = False
                
                if execution_mode.startswith("FastAPI"):
                    try:
                        files = {}
                        if active_media_bytes:
                            files["media_file"] = (active_media_name or "media.png", active_media_bytes, active_media_type)
                        data = {
                            "text": input_text,
                            "enabled_detectors": ",".join(enabled_detector_ids),
                        }
                        res = requests.post(f"{api_url}/analyze", data=data, files=files, timeout=4)
                        if res.status_code == 200:
                            response_data = AnalysisResponse(**res.json())
                        else:
                            fallback_engaged = True
                    except Exception:
                        fallback_engaged = True

                # Automatic fallback when offline or direct mode selected
                if response_data is None:
                    if execution_mode.startswith("FastAPI"):
                        st.info(f"⚡ FastAPI backend at {api_url} is offline or unreachable. Automatically engaged in-memory mock fallback.")
                    
                    response_data = pipeline.run(
                        text=input_text,
                        media_bytes=active_media_bytes,
                        media_filename=active_media_name,
                        media_content_type=active_media_type,
                        enabled_detectors=enabled_detector_ids,
                    )

                if response_data:
                    assessment = response_data.overall_assessment
                    v_class = f"verdict-{assessment.verdict.value}"
                    
                    # Overall Risk Assessment Card (verdict badge, risk level, confidence score)
                    st.markdown(
                        f"""
                        <div class="verdict-card {v_class}">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                <div>
                                    <span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">Overall Verdict</span>
                                    <h2 style="margin: 4px 0 8px 0; font-size: 1.8rem; font-weight: 800;">
                                        {assessment.verdict.value.replace('_', ' ')}
                                    </h2>
                                </div>
                                <div style="text-align: right;">
                                    <span class="badge" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);">
                                        RISK: {assessment.risk_level.value}
                                    </span>
                                    <div style="margin-top: 6px; font-size: 0.9rem; font-weight: 700; color: #38bdf8;">
                                        Confidence: {int(assessment.confidence * 100)}%
                                    </div>
                                </div>
                            </div>
                            <p style="margin: 8px 0 0 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
                                {assessment.summary}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Primary Concerns
                    if assessment.primary_concerns:
                        with st.expander("⚠️ Primary Concerns Flagged", expanded=True):
                            for c in assessment.primary_concerns:
                                st.markdown(f"- {c}")

                    # Detailed inspection tabs
                    tab_findings, tab_evidence, tab_explanation, tab_limitations, tab_json = st.tabs(
                        ["🧩 Findings", "🔬 Evidence", "💡 Rationale", "⚠️ Limitations", "🔌 JSON"]
                    )

                    # Structured Detector Cards (SUCCESS, UNCERTAIN, NOT_RUN, ERROR)
                    with tab_findings:
                        for det_id, finding in response_data.findings.items():
                            status_badge = f'<span class="badge badge-status-{finding.status.value}">{finding.status.value}</span>'
                            score_text = (
                                f"{int(finding.manipulation_score * 100)}%"
                                if finding.manipulation_score is not None
                                else "N/A"
                            )
                            
                            st.markdown(
                                f"""
                                <div class="detector-box">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                        <strong style="font-size: 1.05rem; color: #f8fafc;">{finding.detector_name}</strong>
                                        <div>
                                            {status_badge}
                                        </div>
                                    </div>
                                    <p style="margin: 4px 0 8px 0; color: #cbd5e1; font-size: 0.9rem;">
                                        {finding.summary}
                                    </p>
                                    <div style="display: flex; gap: 20px; font-size: 0.82rem; color: #94a3b8;">
                                        <span>Manipulation Score: <strong style="color: #38bdf8;">{score_text}</strong></span>
                                        <span>Confidence: <strong>{int(finding.confidence * 100)}%</strong></span>
                                        <span>Latency: <strong>{finding.execution_time_ms} ms</strong></span>
                                    </div>
                                </div>
                                "",
                                unsafe_allow_html=True,
                            )
                            if finding.metrics:
                                with st.expander(f"Metrics for {finding.detector_name}", expanded=False):
                                    st.json(finding.metrics)

                    # Concrete Evidence Bullet Points
                    with tab_evidence:
                        if response_data.evidence:
                            for ev in response_data.evidence:
                                sev_color = "#ef4444" if ev.severity.value == "HIGH" else "#f59e0b" if ev.severity.value == "MEDIUM" else "#10b981"
                                st.markdown(
                                    f"""
                                    <div style="border-left: 4px solid {sev_color}; padding: 8px 16px; margin-bottom: 12px; background: rgba(15, 23, 42, 0.4); border-radius: 4px;">
                                        <div style="display: flex; justify-content: space-between;">
                                            <strong style="color: #f1f5f9;">[{ev.evidence_type}] {ev.description}</strong>
                                            <span style="font-size: 0.75rem; color: {sev_color}; font-weight: bold;">{ev.severity.value}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if ev.metadata:
                                    st.caption(f"Evidence Metadata: {json.dumps(ev.metadata)}")
                        else:
                            st.info("No concrete evidentiary anomalies extracted for this input.")

                    # Natural Language Explanation Driven Purely by Active Detector Findings
                    with tab_explanation:
                        st.markdown(f"#### {response_data.explanation.headline}")
                        st.write(response_data.explanation.rationale)
                        
                        col_supp, col_count = st.columns(2)
                        with col_supp:
                            st.markdown("**Supporting Points:**")
                            if response_data.explanation.supporting_points:
                                for sp in response_data.explanation.supporting_points:
                                    st.markdown(f"- {sp}")
                            else:
                                st.caption("No manipulation points flagged.")
                        with col_count:
                            st.markdown("**Counter-Evidence / Baseline Notes:**")
                            if response_data.explanation.counter_evidence:
                                for cp in response_data.explanation.counter_evidence:
                                    st.markdown(f"- {cp}")
                            else:
                                st.caption("No mitigating baseline signals noted.")

                    # Explicit Limitations List
                    with tab_limitations:
                        st.markdown("#### System Governance & Limitations")
                        for disc in response_data.limitations.disclaimers:
                            st.markdown(f"- ℹ️ {disc}")
                        if response_data.limitations.untested_modalities:
                            st.markdown("**Untested Modalities in this Request:**")
                            for um in response_data.limitations.untested_modalities:
                                st.markdown(f"- ⚠️ {um}")
                        if response_data.limitations.model_assumptions:
                            st.markdown("**Model Assumptions:**")
                            for ma in response_data.limitations.model_assumptions:
                                st.markdown(f"- 🔍 {ma}")
                        if response_data.limitations.human_verification_recommended:
                            st.warning("Human-in-the-Loop review is strongly advised prior to definitive enforcement.")

                    # Raw Schema Conformance JSON
                    with tab_json:
                        json_str = response_data.model_dump_json(indent=2)
                        st.code(json_str, language="json")
                        st.download_button(
                            "📥 Download JSON Report",
                            data=json_str,
                            file_name=f"verilens_{response_data.request_id}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
    else:
        st.info("Select a preset from the sidebar or upload custom multimodal content to run verification.")
