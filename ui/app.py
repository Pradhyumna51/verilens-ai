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
                arr[y, x] = [val, (val * 2) % 255, 255 - val]
        img = Image.fromarray(arr)
        img.save(buf, format="PNG")
    elif sample_type == "recycled_news":
        # Create standard landscape image with EXIF mock software
        img = Image.new("RGB", (320, 240), color=(40, 90, 160))
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 280, 200], fill=(70, 130, 200), outline=(255, 255, 255))
        draw.text((60, 110), "HISTORICAL ARCHIVE 2019", fill=(255, 255, 255))
        img.save(buf, format="JPEG")
    elif sample_type == "natural_photo":
        # Smooth photographic gradient
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                arr[y, x] = [int(100 + 100 * (y / 256)), int(120 + 80 * (x / 256)), 180]
        img = Image.fromarray(arr)
        img.save(buf, format="JPEG")
    elif sample_type == "tiny_thumbnail":
        # Under threshold resolution
        img = Image.new("RGB", (20, 20), color=(128, 128, 128))
        img.save(buf, format="PNG")
    elif sample_type == "shark_highway":
        # Flooded asphalt roadway with water gradient
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                arr[y, x] = [int(30 + 40 * (y / 256)), int(70 + 80 * (x / 256)), int(120 + 60 * (y / 256))]
        img = Image.fromarray(arr)
        img.save(buf, format="JPEG")
    return buf.getvalue()

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

    st.markdown("---")
    st.markdown("### 🤖 Cognitive AI Reasoning")
    gemini_key = st.text_input(
        "Gemini API Key (Optional)",
        value=settings.GEMINI_API_KEY or "",
        type="password",
        help="Free key from Google AI Studio (aistudio.google.com). When set, enables Gemini 2.5 Flash multimodal reasoning and explanation generation. When empty, runs fast local offline forensics.",
    )
    if gemini_key:
        st.markdown('<span style="color: #34d399; font-size: 0.85rem; font-weight: 600;">✨ Gemini 2.5 Flash Active (Multimodal)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">🛡️ Local Forensics Ensemble (Offline)</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌐 Fact Check Intelligence")
    fc_key = st.text_input(
        "Google Fact Check API Key (Optional)",
        value=settings.FACT_CHECK_API_KEY or "",
        type="password",
        help="Google Fact Check Tools API key (free from Google Cloud / News Initiative). Enables live cross-referencing against ClaimReview databases (Reuters, Snopes, AP, USA Today, etc.).",
    )
    if fc_key:
        st.markdown('<span style="color: #38bdf8; font-size: 0.85rem; font-weight: 600;">🌐 Google Fact Check Tools Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #94a3b8; font-size: 0.85rem; font-weight: 600;">⚪ Fact Check Query Disabled</span>', unsafe_allow_html=True)

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
            "🌐 6: Live Fact-Check Cross-Reference (Shark on Freeway)",
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
    preset_text = "A serene view of an afternoon lake scene captured during a weekend hike."
    preset_image_bytes = generate_sample_image("natural_photo")
    preset_image_name = "natural_landscape.jpg"
elif preset == "⚠️ 5: Low-Res Thumbnail (UNCERTAIN status)":
    preset_text = "Short text."
    preset_image_bytes = generate_sample_image("tiny_thumbnail")
    preset_image_name = "pixel_thumbnail.png"
elif preset == "🌐 6: Live Fact-Check Cross-Reference (Shark on Freeway)":
    preset_text = "Shark swimming on highway after hurricane"
    preset_image_bytes = generate_sample_image("shark_highway")
    preset_image_name = "shark_flooded_freeway.jpg"

# Main Layout
col_input, col_results = st.columns([1, 1.25], gap="large")

with col_input:
    st.subheader("📥 Multimodal Input")
    
    input_text = st.text_area(
        "Headline / Claim / Caption Text",
        value=preset_text,
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
                
                if execution_mode.startswith("FastAPI"):
                    # Call via REST API
                    try:
                        files = {}
                        if active_media_bytes:
                            files["media_file"] = (active_media_name or "media.png", active_media_bytes, active_media_type)
                        data = {
                            "text": input_text,
                            "enabled_detectors": ",".join(enabled_detector_ids),
                            "api_key": gemini_key or "",
                        }
                        headers = {}
                        if gemini_key:
                            headers["x-gemini-api-key"] = gemini_key
                        if fc_key:
                            headers["x-fact-check-api-key"] = fc_key
                        res = requests.post(f"{api_url}/analyze", data=data, files=files, headers=headers, timeout=25)
                        if res.status_code == 200:
                            response_data = AnalysisResponse(**res.json())
                        else:
                            st.error(f"API Error ({res.status_code}): {res.text}")
                    except Exception as err:
                        st.error(f"Failed to connect to FastAPI at {api_url}: {err}")
                else:
                    # Run direct in-memory pipeline with dynamic module reload
                    import importlib
                    import verilens.core.video
                    import verilens.detectors.synthetic_image
                    import verilens.detectors.deepfake
                    import verilens.detectors.reuse
                    import verilens.detectors.ocr
                    import verilens.pipeline.reasoner
                    import verilens.pipeline.web_knowledge
                    import verilens.pipeline.orchestrator
                    importlib.reload(verilens.core.video)
                    importlib.reload(verilens.detectors.synthetic_image)
                    importlib.reload(verilens.detectors.deepfake)
                    importlib.reload(verilens.detectors.reuse)
                    importlib.reload(verilens.detectors.ocr)
                    importlib.reload(verilens.pipeline.reasoner)
                    importlib.reload(verilens.pipeline.web_knowledge)
                    importlib.reload(verilens.pipeline.orchestrator)
                    active_pipeline = verilens.pipeline.orchestrator.pipeline
                    response_data = active_pipeline.run(
                        text=input_text,
                        media_bytes=active_media_bytes,
                        media_filename=active_media_name,
                        media_content_type=active_media_type,
                        enabled_detectors=enabled_detector_ids,
                        api_key=gemini_key or None,
                        fact_check_api_key=fc_key or None,
                    )

                if response_data:
                    assessment = response_data.overall_assessment
                    v_class = f"verdict-{assessment.verdict.value}"
                    
                    ai_enhanced = any("Gemini" in a for a in response_data.limitations.model_assumptions) or any("gemini" in str(e.metadata) for e in response_data.evidence)
                    engine_badge = (
                        '<span style="display:inline-block; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid #0284c7; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">✨ Powered by Gemini 2.5 Flash Multimodal Forensics</span>'
                        if ai_enhanced
                        else '<span style="display:inline-block; background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid #64748b; padding: 4px 12px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">🛡️ Powered by Local Forensic Ensemble</span>'
                    )

                    basis_html = ""
                    if getattr(assessment, "basis_sources", None):
                        items_html = "".join([f"<li style='margin-bottom: 4px; color: #cbd5e1;'>{s}</li>" for s in assessment.basis_sources])
                        basis_html = (
                            '<div style="margin-top: 14px; padding: 12px 14px; background: rgba(15, 23, 42, 0.75); border-radius: 8px; border: 1px solid rgba(255,255,255,0.08);">'
                            '<div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">'
                            '🔍 Verified Basis & Evidence Sources'
                            '</div>'
                            f'<ul style="margin: 0; padding-left: 18px; font-size: 0.85rem; line-height: 1.45;">{items_html}</ul>'
                            '</div>'
                        )

                    # Verdict Card
                    card_html = (
                        f'<div class="verdict-card {v_class}">'
                        f'<div style="display: flex; justify-content: space-between; align-items: flex-start;">'
                        f'<div>'
                        f'<span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase;">Overall Verdict</span>'
                        f'<h2 style="margin: 4px 0 8px 0; font-size: 1.8rem; font-weight: 800;">{assessment.verdict.value.replace("_", " ")}</h2>'
                        f'</div>'
                        f'<div style="text-align: right;">'
                        f'<span class="badge" style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);">RISK: {assessment.risk_level.value}</span>'
                        f'<div style="margin-top: 6px; font-size: 0.9rem; font-weight: 700; color: #38bdf8;">Confidence: {int(assessment.confidence * 100)}%</div>'
                        f'</div>'
                        f'</div>'
                        f'<p style="margin: 8px 0 0 0; color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">{assessment.summary}</p>'
                        f'{basis_html}'
                        f'<div style="margin-top: 12px;">{engine_badge}</div>'
                        f'</div>'
                    )
                    st.markdown(card_html, unsafe_allow_html=True)

                    # Verified Fact-Check Banner (if ClaimReview debunks found)
                    verified_fcs = [e for e in response_data.evidence if e.evidence_type == "VERIFIED_FACT_CHECK"]
                    if verified_fcs:
                        direct_matches = [e for e in verified_fcs if (e.metadata or {}).get("is_direct_match", True)]
                        banner_title = "Direct Verified Fact-Check Debunk Found" if direct_matches else "Related Fact-Check Topic Context"
                        banner_border = "#ef4444" if direct_matches else "#0ea5e9"
                        banner_bg = "rgba(239, 68, 68, 0.12)" if direct_matches else "rgba(14, 165, 233, 0.12)"
                        banner_icon = "🚨" if direct_matches else "🌐"

                        st.markdown(
                            f"""
                            <div style="background: {banner_bg}; border: 1px solid {banner_border}; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                                    <span style="font-size: 1.25rem;">{banner_icon}</span>
                                    <strong style="color: #f8fafc; font-size: 1.05rem;">{banner_title} (Google Fact Check Tools)</strong>
                                </div>
                                <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0 0 12px 0;">
                                    {'This input claim directly corresponds to a verified debunk in ClaimReview archives:' if direct_matches else 'Independent fact-checkers investigated the following related rumors on this general topic (note: distinct from your input claim):'}
                                </p>
                            """,
                            unsafe_allow_html=True,
                        )
                        for fc in verified_fcs:
                            meta = fc.metadata or {}
                            pub = meta.get("publisher", "Verified Fact-Checker")
                            rating = meta.get("rating", "Unrated")
                            url = meta.get("url", "")
                            claim = meta.get("claim", "")
                            is_direct = meta.get("is_direct_match", True)
                            date_val = meta.get("review_date", "")
                            date_str = f" &bull; <span style='color: #64748b;'>{date_val[:10]}</span>" if date_val else ""
                            link_markup = f'<a href="{url}" target="_blank" style="color: #38bdf8; text-decoration: underline; font-weight: 600; font-size: 0.85rem;">Read Verified Source Article ↗</a>' if url else ''
                            
                            badge_bg = "rgba(239, 68, 68, 0.2)" if is_direct else "rgba(56, 189, 248, 0.15)"
                            badge_color = "#f87171" if is_direct else "#38bdf8"
                            badge_border = "#ef4444" if is_direct else "#0284c7"
                            badge_label = f"DIRECT MATCH &bull; RATING: {rating}" if is_direct else f"RELATED TOPIC &bull; RATING: {rating}"
                            note_markup = "" if is_direct else '<div style="color: #94a3b8; font-size: 0.78rem; font-style: italic; margin-top: 4px;">ℹ️ Fact-checker evaluated a different rumor on this general topic, not your specific statement.</div>'

                            st.markdown(
                                f"""
                                <div style="background: rgba(15, 23, 42, 0.65); border-left: 4px solid {'#ef4444' if is_direct else '#38bdf8'}; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <strong style="color: #f8fafc; font-size: 0.95rem;">{pub}{date_str}</strong>
                                        <span style="background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; border-radius: 9999px; padding: 2px 10px; font-weight: 700; font-size: 0.78rem;">
                                            {badge_label}
                                        </span>
                                    </div>
                                    <div style="color: #cbd5e1; font-size: 0.85rem; margin: 6px 0;">Claim Reviewed: &ldquo;{claim}&rdquo;</div>
                                    <div>{link_markup}</div>
                                    {note_markup}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Primary Concerns
                    if assessment.primary_concerns:
                        with st.expander("⚠️ Primary Concerns Flagged", expanded=True):
                            for c in assessment.primary_concerns:
                                st.markdown(f"- {c}")

                    # Tabs for detailed inspection
                    tab_findings, tab_evidence, tab_visualizer, tab_explanation, tab_limitations, tab_json = st.tabs(
                        ["🧩 Findings", "🔬 Evidence", "🖼️ Forensic Visualizer", "💡 Rationale", "⚠️ Limitations", "🔌 JSON"]
                    )

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
                                """,
                                unsafe_allow_html=True,
                            )
                            if finding.metrics:
                                with st.expander(f"Metrics for {finding.detector_name}", expanded=False):
                                    st.json(finding.metrics)

                    with tab_evidence:
                        if response_data.evidence:
                            for ev in response_data.evidence:
                                sev_color = "#ef4444" if ev.severity.value == "HIGH" else "#f59e0b" if ev.severity.value == "MEDIUM" else "#10b981"
                                icon = "🌐" if ev.evidence_type == "VERIFIED_FACT_CHECK" else "🔬"
                                st.markdown(
                                    f"""
                                    <div style="border-left: 4px solid {sev_color}; padding: 8px 16px; margin-bottom: 12px; background: rgba(15, 23, 42, 0.4); border-radius: 4px;">
                                        <div style="display: flex; justify-content: space-between;">
                                            <strong style="color: #f1f5f9;">{icon} [{ev.evidence_type}] {ev.description}</strong>
                                            <span style="font-size: 0.75rem; color: {sev_color}; font-weight: bold;">{ev.severity.value}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if ev.metadata:
                                    if ev.evidence_type == "VERIFIED_FACT_CHECK" and ev.metadata.get("url"):
                                        st.markdown(f"🔗 **Fact-Check Article:** [{ev.metadata.get('url')}]({ev.metadata.get('url')})")
                                    else:
                                        st.caption(f"Evidence Metadata: {json.dumps(ev.metadata)}")
                        else:
                            st.info("No concrete evidentiary anomalies extracted for this input.")

                    with tab_visualizer:
                        st.markdown("#### 🔬 Multi-Modal Forensic Visualizer (MVNN & DIDAN Insights)")
                        if active_media_bytes and "image" in active_media_type:
                            try:
                                orig = Image.open(io.BytesIO(active_media_bytes)).convert("RGB")
                                buf_ela = io.BytesIO()
                                orig.save(buf_ela, format="JPEG", quality=90)
                                buf_ela.seek(0)
                                recomp = Image.open(buf_ela).convert("RGB")
                                diff = np.abs(np.asarray(orig, dtype=np.int16) - np.asarray(recomp, dtype=np.int16))
                                scale_diff = np.clip(diff * 12, 0, 255).astype(np.uint8)
                                ela_img = Image.fromarray(scale_diff)

                                col_orig, col_ela = st.columns(2)
                                with col_orig:
                                    st.markdown("##### 📷 Original Uploaded Media")
                                    st.image(orig, use_container_width=True)
                                with col_ela:
                                    st.markdown("##### 🔬 Error Level Analysis (ELA Heatmap)")
                                    st.image(ela_img, use_container_width=True)
                                    st.caption("Amplified pixel compression residuals (MVNN Pixel Domain). Spliced or in-painted regions stand out as high-intensity error clusters.")
                            except Exception as ex:
                                st.warning(f"Could not render ELA visualization: {ex}")
                        else:
                            st.info("Upload or select an image to inspect visual domain Error Level Analysis (ELA).")

                        # Forensics Telemetry Cards
                        syn_img_metrics = (response_data.findings.get("synthetic_image") or {}).metrics or {}
                        syn_txt_metrics = (response_data.findings.get("synthetic_text") or {}).metrics or {}
                        consist_metrics = (response_data.findings.get("consistency") or {}).metrics or {}

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("2D FFT High-Freq Ratio", syn_img_metrics.get("high_frequency_ratio", "N/A"))
                        with c2:
                            st.metric("ELA Max Error", syn_img_metrics.get("ela_max_error", "N/A"))
                        with c3:
                            st.metric("Text Emotional Arousal", syn_txt_metrics.get("emotional_arousal_score", "N/A"))
                        with c4:
                            st.metric("DIDAN Discrepancy", consist_metrics.get("didan_discrepancy_score", "N/A"))

                    with tab_explanation:
                        st.markdown(f"#### {response_data.explanation.headline}")
                        st.write(response_data.explanation.rationale)
                        
                        col_supp, col_count = st.columns(2)
                        with col_supp:
                            st.markdown("**Supporting Points:**")
                            for sp in response_data.explanation.supporting_points:
                                st.markdown(f"- {sp}")
                        with col_count:
                            st.markdown("**Counter-Evidence / Baseline Notes:**")
                            for cp in response_data.explanation.counter_evidence:
                                st.markdown(f"- {cp}")

                    with tab_limitations:
                        st.markdown("#### System Governance & Limitations")
                        for disc in response_data.limitations.disclaimers:
                            st.markdown(f"- ℹ️ {disc}")
                        if response_data.limitations.untested_modalities:
                            st.markdown("**Untested Modalities in this Request:**")
                            for um in response_data.limitations.untested_modalities:
                                st.markdown(f"- ⚠️ {um}")
                        if response_data.limitations.human_verification_recommended:
                            st.warning("Human-in-the-Loop review is strongly advised prior to definitive enforcement.")

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
