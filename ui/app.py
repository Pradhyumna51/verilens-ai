"""
VeriLens AI - Multimodal Forensic Intelligence Engine
Anti-Slop Enterprise Architecture (design-taste-frontend compliant)
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

# -----------------------------------------------------------------------------
# HTML Helper: Strips indentation to prevent markdown from generating code blocks
# -----------------------------------------------------------------------------
def render_html(raw_html: str) -> None:
    """Renders HTML in Streamlit ensuring no leading indentation triggers markdown code blocks."""
    cleaned = " ".join(line.strip() for line in raw_html.split("\n") if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriLens AI - Multimodal Forensic Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Anti-Slop Design System (CSS)
# Neutral graphite base, zero glowing gradients, zero cartoon emojis, crisp type
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

    :root {
        --canvas-bg: #0d1117;
        --surface-bg: #161b22;
        --surface-raised: #21262d;
        --border-default: #30363d;
        --border-subtle: #21262d;
        --text-primary: #f0f6fc;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-blue: #1f6feb;
        --accent-green: #238636;
        --accent-red: #da3633;
        --accent-amber: #d29922;
        --font-stack: "Geist", -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
        --font-mono: "Geist Mono", ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    }

    /* Suppress Streamlit Default Header Rainbow Bar and Footer */
    [data-testid="stDecoration"] {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    *, *::before, *::after,
    html, body, .stApp,
    p, span, div, h1, h2, h3, h4, h5, h6,
    label, button, input, textarea, select {
        font-family: var(--font-stack) !important;
        -webkit-font-smoothing: antialiased;
    }

    code, pre, .font-mono, [data-testid="stMetricValue"], .stat-pill, .detector-metrics {
        font-family: var(--font-mono) !important;
    }

    .stApp {
        background: var(--canvas-bg) !important;
        color: var(--text-primary) !important;
    }

    /* Sidebar: Clean Dark Drawer */
    [data-testid="stSidebar"] {
        background: var(--surface-bg) !important;
        border-right: 1px solid var(--border-default) !important;
    }

    /* Minimal Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--border-default);
    }
    .app-title {
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text-primary);
        margin: 0;
    }
    .app-meta {
        font-size: 0.82rem;
        color: var(--text-secondary);
        margin: 4px 0 0 0;
    }
    .status-indicator {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 600;
        color: #3fb950;
        background: rgba(46, 160, 67, 0.12);
        border: 1px solid rgba(46, 160, 67, 0.3);
        padding: 4px 10px;
        border-radius: 4px;
    }

    /* Section Headers */
    .section-title {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 12px;
    }

    /* Verdict Card (Authoritative, No Glowing Blobs) */
    .verdict-card {
        background: var(--surface-bg);
        border: 1px solid var(--border-default);
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    .verdict-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 12px;
    }
    .verdict-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }
    .verdict-headline {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 4px 0 0 0;
    }

    .v-badge-LIKELY_MANIPULATED {
        color: #f85149;
    }
    .v-badge-SUSPICIOUS {
        color: #d29922;
    }
    .v-badge-LIKELY_REAL {
        color: #3fb950;
    }
    .v-badge-UNCERTAIN {
        color: #a371f7;
    }

    .stats-strip {
        display: flex;
        gap: 16px;
        align-items: center;
        text-align: right;
    }
    .stat-pill {
        font-family: var(--font-mono);
        font-size: 0.78rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid var(--border-default);
        background: var(--surface-raised);
        color: var(--text-secondary);
    }

    .verdict-summary {
        font-size: 0.92rem;
        line-height: 1.55;
        color: var(--text-primary);
        margin: 0 0 16px 0;
    }

    /* Basis and Evidence Tray */
    .basis-box {
        background: var(--canvas-bg);
        border: 1px solid var(--border-default);
        border-radius: 4px;
        padding: 14px 18px;
        margin-top: 14px;
    }
    .basis-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }
    .basis-items {
        margin: 0;
        padding-left: 18px;
        font-size: 0.84rem;
        line-height: 1.6;
        color: var(--text-primary);
    }

    /* Findings Cards */
    .detector-row {
        background: var(--surface-bg);
        border: 1px solid var(--border-default);
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .detector-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .detector-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .detector-summary {
        font-size: 0.85rem;
        line-height: 1.45;
        color: var(--text-secondary);
        margin: 0 0 10px 0;
    }
    .detector-metrics {
        display: flex;
        gap: 20px;
        font-family: var(--font-mono);
        font-size: 0.78rem;
        color: var(--text-muted);
    }
    .detector-metrics strong {
        color: var(--text-primary);
    }

    /* Status Badges */
    .badge-status {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 3px;
        text-transform: uppercase;
    }
    .badge-SUCCESS { background: rgba(46, 160, 67, 0.15); color: #3fb950; border: 1px solid rgba(46, 160, 67, 0.35); }
    .badge-UNCERTAIN { background: rgba(210, 153, 34, 0.15); color: #d29922; border: 1px solid rgba(210, 153, 34, 0.35); }
    .badge-ERROR { background: rgba(248, 81, 73, 0.15); color: #f85149; border: 1px solid rgba(248, 81, 73, 0.35); }
    .badge-NOT_RUN { background: rgba(110, 118, 129, 0.12); color: #8b949e; border: 1px solid rgba(110, 118, 129, 0.25); }

    /* Telemetry Metric Cards */
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 14px;
    }
    .telemetry-card {
        background: var(--surface-bg);
        border: 1px solid var(--border-default);
        border-radius: 4px;
        padding: 12px;
        text-align: center;
    }
    .telemetry-caption {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .telemetry-number {
        font-family: var(--font-mono);
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-top: 4px;
    }

    /* Primary Action Button: Linear / GitHub Style */
    div.stButton > button:first-child {
        background: var(--accent-green) !important;
        color: #ffffff !important;
        border: 1px solid rgba(240, 246, 252, 0.1) !important;
        border-radius: 6px !important;
        padding: 9px 18px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2) !important;
        cursor: pointer !important;
        transition: background 0.15s ease, transform 0.05s ease !important;
    }
    div.stButton > button:first-child:hover {
        background: #2ea043 !important;
    }
    div.stButton > button:first-child:active {
        background: #238636 !important;
        transform: translateY(1px);
    }

    /* Secondary / Download Buttons */
    .stDownloadButton > button {
        background: var(--surface-raised) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: background 0.15s ease !important;
    }
    .stDownloadButton > button:hover {
        background: var(--border-default) !important;
        border-color: #8b949e !important;
    }

    /* Inputs: Dark Matte */
    div[data-baseweb="textarea"] {
        background: var(--canvas-bg) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="textarea"]:focus-within {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 1px var(--accent-blue) !important;
    }

    [data-testid="stFileUploader"] {
        background: var(--canvas-bg);
        border: 1px dashed var(--border-default);
        border-radius: 6px;
        padding: 12px;
    }

    /* Expanders: Clean Bordered Container */
    [data-testid="stExpander"] {
        background: var(--surface-bg) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 6px !important;
        margin-bottom: 12px !important;
    }

    /* Tabs: Understated Border Line */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border-default) !important;
        padding: 0 !important;
        gap: 20px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border: none !important;
        padding: 8px 4px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--accent-blue) !important;
        font-weight: 600 !important;
    }

    /* Alerts: Calibrated Dark Surface */
    div[data-testid="stAlert"] {
        background: var(--surface-bg) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }

    /* Fact Check Card */
    .factcheck-container {
        background: var(--surface-bg);
        border: 1px solid var(--border-default);
        border-left: 3px solid var(--accent-blue);
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 16px;
    }
    .factcheck-direct {
        border-left-color: var(--accent-red);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Test Image Generation Helper for Presets
# -----------------------------------------------------------------------------
def generate_sample_image(sample_type: str) -> bytes:
    buf = io.BytesIO()
    if sample_type == "synthetic_grid":
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                val = int((np.sin(x / 2.0) * np.cos(y / 2.0) + 1.0) * 127)
                arr[y, x] = [val, (val * 2) % 255, 255 - val]
        img = Image.fromarray(arr)
        img.save(buf, format="PNG")
    elif sample_type == "recycled_news":
        img = Image.new("RGB", (320, 240), color=(40, 90, 160))
        draw = ImageDraw.Draw(img)
        draw.rectangle([40, 40, 280, 200], fill=(70, 130, 200), outline=(255, 255, 255))
        draw.text((60, 110), "HISTORICAL ARCHIVE 2019", fill=(255, 255, 255))
        img.save(buf, format="JPEG")
    elif sample_type == "natural_photo":
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                arr[y, x] = [int(100 + 100 * (y / 256)), int(120 + 80 * (x / 256)), 180]
        img = Image.fromarray(arr)
        img.save(buf, format="JPEG")
    elif sample_type == "tiny_thumbnail":
        img = Image.new("RGB", (20, 20), color=(128, 128, 128))
        img.save(buf, format="PNG")
    elif sample_type == "shark_highway":
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                arr[y, x] = [int(30 + 40 * (y / 256)), int(70 + 80 * (x / 256)), int(120 + 60 * (y / 256))]
        img = Image.fromarray(arr)
        img.save(buf, format="JPEG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# App Header (Restrained, Professional)
# -----------------------------------------------------------------------------
render_html(
    """
    <div class="app-header">
        <div>
            <h1 class="app-title">VeriLens AI</h1>
            <p class="app-meta">Multimodal Misinformation Detection & Deepfake Forensic Intelligence</p>
        </div>
        <div>
            <span class="status-indicator">SYSTEM READY</span>
        </div>
    </div>
    """
)

# -----------------------------------------------------------------------------
# Sidebar: System Configuration (Clean, No Emojis, No Exposed Keys)
# -----------------------------------------------------------------------------
gemini_key = settings.GEMINI_API_KEY or None
fc_key = settings.FACT_CHECK_API_KEY or None

with st.sidebar:
    st.markdown("<div class='section-title'>Configuration</div>", unsafe_allow_html=True)

    execution_mode = st.radio(
        "Execution Mode",
        ["Direct Orchestrator (In-Memory)", "FastAPI REST API (http://127.0.0.1:8000)"],
        index=0,
    )
    api_url = st.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")

    st.markdown("---")
    st.markdown("<div class='section-title'>Evaluation Presets</div>", unsafe_allow_html=True)
    
    preset = st.selectbox(
        "Preset Scenario",
        [
            "Custom Input",
            "1. Synthetic Diffusion (Frequency Anomaly)",
            "2. Recycled Historical Media (Hash Collision)",
            "3. Synthetic Text (Burstiness Test)",
            "4. Camera Capture (Organic Baseline)",
            "5. Low-Resolution Media (Boundary Test)",
            "6. Verified Debunk Registry (Highway Shark)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div class='section-title'>Active Detectors</div>", unsafe_allow_html=True)
    available_detectors = registry.list_all()
    enabled_detector_ids = []
    for d in available_detectors:
        checked = st.checkbox(d.detector_name, value=True, key=f"det_{d.detector_id}")
        if checked:
            enabled_detector_ids.append(d.detector_id)

    st.markdown("---")
    st.caption(f"Payload Limits: {settings.MAX_MEDIA_SIZE_BYTES // (1024*1024)} MB media &bull; {settings.MAX_TEXT_CHARS:,} chars text")

# -----------------------------------------------------------------------------
# Preset Scenario Auto-Population
# -----------------------------------------------------------------------------
preset_text = ""
preset_image_bytes = None
preset_image_name = None

if preset == "1. Synthetic Diffusion (Frequency Anomaly)":
    preset_text = "Breaking News: Dangerous firestorm engulfs the downtown metropolitan subway station during midnight hours."
    preset_image_bytes = generate_sample_image("synthetic_grid")
    preset_image_name = "synthetic_diffusion_sample.png"
elif preset == "2. Recycled Historical Media (Hash Collision)":
    preset_text = "Urgent: Current ongoing coastal disaster reported just 10 minutes ago."
    preset_image_bytes = generate_sample_image("recycled_news")
    preset_image_name = "recycled_archive_2019.jpg"
elif preset == "3. Synthetic Text (Burstiness Test)":
    preset_text = (
        "In conclusion, it is important to remember that modern technological frameworks delve into a rich tapestry of innovation. "
        "Furthermore, artificial intelligence plays a crucial role across interconnected systems. "
        "Moreover, digital infrastructure serves as a profound testament to ongoing scientific evolution. "
        "Therefore, one must appreciate the multifaceted dynamics governing our contemporary algorithmic reality."
    )
elif preset == "4. Camera Capture (Organic Baseline)":
    preset_text = "A serene view of an afternoon lake scene captured during a weekend hike."
    preset_image_bytes = generate_sample_image("natural_photo")
    preset_image_name = "natural_landscape.jpg"
elif preset == "5. Low-Resolution Media (Boundary Test)":
    preset_text = "Short text."
    preset_image_bytes = generate_sample_image("tiny_thumbnail")
    preset_image_name = "pixel_thumbnail.png"
elif preset == "6. Verified Debunk Registry (Highway Shark)":
    preset_text = "Shark swimming on highway after hurricane"
    preset_image_bytes = generate_sample_image("shark_highway")
    preset_image_name = "shark_flooded_freeway.jpg"

# -----------------------------------------------------------------------------
# Main Application Layout: Two-Column Workbench
# -----------------------------------------------------------------------------
col_input, col_results = st.columns([1, 1.25], gap="large")

with col_input:
    st.markdown("<div class='section-title'>Input Modalities</div>", unsafe_allow_html=True)
    
    input_text = st.text_area(
        "Claim Text or Caption",
        value=preset_text,
        height=140,
        placeholder="Enter headline, social media caption, or claim to verify...",
        help="Analyzed for synthetic stylometry, cross-modal contradictions, and verified ClaimReview debunks.",
    )
    
    uploaded_file = st.file_uploader(
        "Media Asset (Image or Video)",
        type=["png", "jpg", "jpeg", "webp", "mp4", "mov"],
        help=f"Supported formats: JPG, PNG, WEBP, MP4, MOV up to {settings.MAX_MEDIA_SIZE_BYTES // (1024*1024)}MB",
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
        st.image(active_media_bytes, caption=f"Loaded: {active_media_name}", use_container_width=True)
    elif active_media_bytes and active_media_type and active_media_type.startswith("video/"):
        st.video(active_media_bytes)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("Run Verification", type="primary", use_container_width=True)

with col_results:
    st.markdown("<div class='section-title'>Verification Dossier</div>", unsafe_allow_html=True)
    
    if analyze_btn:
        if not input_text and not active_media_bytes:
            st.warning("Please supply at least one input modality (text or media file).")
        else:
            with st.spinner("Analyzing active forensic channels and cross-referencing knowledge bases..."):
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
                    v_key = assessment.verdict.value
                    
                    # Clean Risk Color
                    risk_color = (
                        "#f85149" if assessment.risk_level.value == "CRITICAL"
                        else "#f85149" if assessment.risk_level.value == "HIGH"
                        else "#d29922" if assessment.risk_level.value == "MEDIUM"
                        else "#3fb950"
                    )

                    # Basis Tray (Flat contiguous HTML)
                    basis_html = ""
                    if getattr(assessment, "basis_sources", None):
                        items_html = "".join([f"<li>{s}</li>" for s in assessment.basis_sources])
                        basis_html = (
                            "<div class='basis-box'>"
                            "<div class='basis-title'>Evidence Sources & Findings Basis</div>"
                            f"<ul class='basis-items'>{items_html}</ul>"
                            "</div>"
                        )

                    # Verdict Card (Clean, authoritative typography)
                    verdict_display = assessment.verdict.value.replace("_", " ")
                    card_html = (
                        "<div class='verdict-card'>"
                        "<div class='verdict-header'>"
                        "<div>"
                        "<div class='verdict-label'>Overall Verdict</div>"
                        f"<div class='verdict-headline v-badge-{v_key}'>{verdict_display}</div>"
                        "</div>"
                        "<div class='stats-strip'>"
                        f"<span class='stat-pill' style='color: {risk_color}; border-color: {risk_color}44;'>Risk: {assessment.risk_level.value}</span>"
                        f"<span class='stat-pill'>Confidence: {int(assessment.confidence * 100)}%</span>"
                        f"<span class='stat-pill'>{response_data.processing_time_ms:.0f} ms</span>"
                        "</div>"
                        "</div>"
                        f"<p class='verdict-summary'>{assessment.summary}</p>"
                        f"{basis_html}"
                        "</div>"
                    )
                    render_html(card_html)

                    # Verified Fact-Check Debunk Callout (if ClaimReview debunks found)
                    verified_fcs = [e for e in response_data.evidence if e.evidence_type == "VERIFIED_FACT_CHECK"]
                    if verified_fcs:
                        direct_matches = [e for e in verified_fcs if (e.metadata or {}).get("is_direct_match", True)]
                        banner_title = "Direct Verified Fact-Check Match" if direct_matches else "Related Fact-Check Record"

                        fc_items = ""
                        for fc in verified_fcs:
                            meta = fc.metadata or {}
                            pub = meta.get("publisher", "ClaimReview Publisher")
                            rating = meta.get("rating", "Unrated")
                            url = meta.get("url", "")
                            claim = meta.get("claim", "")
                            is_direct = meta.get("is_direct_match", True)
                            date_val = meta.get("review_date", "")
                            date_str = f" - {date_val[:10]}" if date_val else ""
                            link_markup = f"<a href='{url}' target='_blank' style='color: #58a6ff; font-size: 0.82rem;'>Read Source Article</a>" if url else ""
                            badge_color = "#f85149" if is_direct else "#58a6ff"
                            badge_text = f"DIRECT - {rating}" if is_direct else f"TOPIC - {rating}"

                            fc_items += (
                                f"<div style='margin-top: 10px; padding: 10px 14px; background: #161b22; border: 1px solid #30363d; border-radius: 4px;'>"
                                f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                                f"<strong style='font-size: 0.88rem; color: #f0f6fc;'>{pub}{date_str}</strong>"
                                f"<span style='font-family: var(--font-mono); font-size: 0.72rem; color: {badge_color}; border: 1px solid {badge_color}44; padding: 2px 6px; border-radius: 3px;'>{badge_text}</span>"
                                f"</div>"
                                f"<div style='color: #8b949e; font-size: 0.82rem; margin: 4px 0;'>\"{claim}\"</div>"
                                f"<div>{link_markup}</div>"
                                f"</div>"
                            )

                        fc_banner_html = (
                            f"<div class='factcheck-container {'factcheck-direct' if direct_matches else ''}'>"
                            f"<div style='font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: {'#f85149' if direct_matches else '#58a6ff'};'>{banner_title}</div>"
                            f"<p style='color: #8b949e; font-size: 0.82rem; margin: 4px 0 0 0;'>Independent fact-checkers investigated claims corresponding to this submission:</p>"
                            f"{fc_items}"
                            f"</div>"
                        )
                        render_html(fc_banner_html)

                    # Primary Concerns
                    if assessment.primary_concerns:
                        with st.expander("Flagged Anomalies", expanded=True):
                            for c in assessment.primary_concerns:
                                st.markdown(f"- {c}")

                    # Tabs: Clean, Text-Only
                    tab_findings, tab_evidence, tab_visualizer, tab_explanation, tab_limitations, tab_json = st.tabs(
                        ["Findings", "Evidence", "Forensic Visualizer", "Rationale", "Governance", "JSON"]
                    )

                    with tab_findings:
                        for det_id, finding in response_data.findings.items():
                            status_badge = f"<span class='badge-status badge-{finding.status.value}'>{finding.status.value}</span>"
                            score_text = (
                                f"{int(finding.manipulation_score * 100)}%"
                                if finding.manipulation_score is not None
                                else "N/A"
                            )
                            score_val = finding.manipulation_score or 0.0
                            score_color = "#f85149" if score_val >= 0.70 else "#d29922" if score_val >= 0.45 else "#3fb950"

                            row_html = (
                                "<div class='detector-row'>"
                                "<div class='detector-header'>"
                                f"<span class='detector-name'>{finding.detector_name}</span>"
                                f"{status_badge}"
                                "</div>"
                                f"<p class='detector-summary'>{finding.summary}</p>"
                                "<div class='detector-metrics'>"
                                f"<span>Manipulation: <strong style='color: {score_color};'>{score_text}</strong></span>"
                                f"<span>Confidence: <strong>{int(finding.confidence * 100)}%</strong></span>"
                                f"<span>Latency: <strong>{finding.execution_time_ms} ms</strong></span>"
                                "</div>"
                                "</div>"
                            )
                            render_html(row_html)
                            if finding.metrics:
                                with st.expander(f"Metrics: {finding.detector_name}", expanded=False):
                                    st.json(finding.metrics)

                    with tab_evidence:
                        if response_data.evidence:
                            for ev in response_data.evidence:
                                sev_color = "#f85149" if ev.severity.value == "HIGH" else "#d29922" if ev.severity.value == "MEDIUM" else "#3fb950"
                                ev_html = (
                                    f"<div style='border-left: 3px solid {sev_color}; padding: 8px 14px; margin-bottom: 8px; background: #161b22; border-radius: 3px;'>"
                                    f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                                    f"<span style='color: #f0f6fc; font-size: 0.86rem;'>[{ev.evidence_type}] {ev.description}</span>"
                                    f"<span style='font-family: var(--font-mono); font-size: 0.72rem; color: {sev_color}; font-weight: 600;'>{ev.severity.value}</span>"
                                    f"</div>"
                                    f"</div>"
                                )
                                render_html(ev_html)
                                if ev.metadata and ev.metadata.get("url"):
                                    st.caption(f"Source URL: {ev.metadata.get('url')}")
                        else:
                            st.info("No explicit forensic anomaly items registered.")

                    with tab_visualizer:
                        st.markdown("<div class='section-title'>Frequency & Compression Residuals</div>", unsafe_allow_html=True)
                        if active_media_bytes and "image" in str(active_media_type):
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
                                    st.caption("Input Image")
                                    st.image(orig, use_container_width=True)
                                with col_ela:
                                    st.caption("Error Level Analysis (Compression Residuals)")
                                    st.image(ela_img, use_container_width=True)
                            except Exception as ex:
                                st.warning(f"Could not render ELA visualization: {ex}")
                        else:
                            st.info("Upload an image asset to inspect visual Error Level Analysis (ELA).")

                        # Telemetry Grid (Data-dense, Clean)
                        syn_img_metrics = (response_data.findings.get("synthetic_image") or {}).metrics or {}
                        syn_txt_metrics = (response_data.findings.get("synthetic_text") or {}).metrics or {}
                        consist_metrics = (response_data.findings.get("consistency") or {}).metrics or {}
                        df_metrics = (response_data.findings.get("deepfake") or {}).metrics or {}

                        t1, t2, t3, t4 = st.columns(4)
                        with t1:
                            render_html(
                                f"<div class='telemetry-card'><div class='telemetry-caption'>2D FFT High-Freq</div><div class='telemetry-number'>{syn_img_metrics.get('high_frequency_ratio', 'N/A')}</div></div>"
                            )
                        with t2:
                            render_html(
                                f"<div class='telemetry-card'><div class='telemetry-caption'>ELA Residual</div><div class='telemetry-number'>{syn_img_metrics.get('ela_max_error', 'N/A')}</div></div>"
                            )
                        with t3:
                            render_html(
                                f"<div class='telemetry-card'><div class='telemetry-caption'>Warping Delta</div><div class='telemetry-number'>{df_metrics.get('morphological_warping', df_metrics.get('temporal_face_jitter', 'N/A'))}</div></div>"
                            )
                        with t4:
                            render_html(
                                f"<div class='telemetry-card'><div class='telemetry-caption'>Semantic Delta</div><div class='telemetry-number'>{consist_metrics.get('didan_discrepancy_score', 'N/A')}</div></div>"
                            )

                    with tab_explanation:
                        st.markdown(f"**{response_data.explanation.headline}**")
                        st.write(response_data.explanation.rationale)
                        
                        col_supp, col_count = st.columns(2)
                        with col_supp:
                            st.markdown("**Supporting Points**")
                            for sp in response_data.explanation.supporting_points:
                                st.markdown(f"- {sp}")
                        with col_count:
                            st.markdown("**Counter-Evidence**")
                            for cp in response_data.explanation.counter_evidence:
                                st.markdown(f"- {cp}")

                    with tab_limitations:
                        st.markdown("**System Governance & Limitations**")
                        for disc in response_data.limitations.disclaimers:
                            st.markdown(f"- {disc}")
                        if response_data.limitations.untested_modalities:
                            st.markdown("**Untested Modalities**")
                            for um in response_data.limitations.untested_modalities:
                                st.markdown(f"- {um}")
                        if response_data.limitations.human_verification_recommended:
                            st.info("Human-in-the-loop review recommended prior to definitive enforcement.")

                    with tab_json:
                        json_str = response_data.model_dump_json(indent=2)
                        st.code(json_str, language="json")
                        st.download_button(
                            "Export JSON Dossier",
                            data=json_str,
                            file_name=f"verilens_{response_data.request_id}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
    else:
        st.info("Select a preset scenario or provide media and text inputs to run verification.")
