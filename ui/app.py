"""
VeriLens AI - Multimodal Misinformation & Forensic Verification Engine
Crafted with Apple-tier Fluid Dynamics, High-End Visual Design & Doppelrand Architecture
Enterprise Production Edition
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
# HTML Rendering Helper (Guarantees zero markdown code block indentation parsing)
# -----------------------------------------------------------------------------
def render_html(raw_html: str) -> None:
    """Renders HTML in Streamlit ensuring no leading indentation triggers markdown code blocks."""
    cleaned = " ".join(line.strip() for line in raw_html.split("\n") if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VeriLens AI | Multimodal Forensic Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Apple-Tier Design System & High-End Visual Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Plus Jakarta Sans", sans-serif;
        --font-mono: "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
        --spring-curve: cubic-bezier(0.16, 1, 0.3, 1);
        --ease-smooth: cubic-bezier(0.4, 0, 0.2, 1);
        --card-shell: rgba(255, 255, 255, 0.035);
        --card-shell-border: rgba(255, 255, 255, 0.075);
        --card-inner: rgba(13, 17, 28, 0.78);
        --accent-cyan: #38bdf8;
        --accent-indigo: #6366f1;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans);
        color: #f1f5f9;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }

    /* Ambient Canvas Mesh Gradient */
    .stApp {
        background: #07090e;
        background-image: 
            radial-gradient(100% 70% at 50% -10%, rgba(56, 189, 248, 0.09) 0%, rgba(99, 102, 241, 0.04) 40%, transparent 80%),
            radial-gradient(60% 50% at 90% 85%, rgba(16, 185, 129, 0.03) 0%, transparent 100%),
            radial-gradient(50% 50% at 10% 90%, rgba(244, 63, 94, 0.03) 0%, transparent 100%);
        background-attachment: fixed;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 23, 0.82) !important;
        backdrop-filter: blur(28px) saturate(190%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }

    /* Floating Island Header */
    .apple-nav-header {
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 18px 28px;
        margin-bottom: 28px;
        background: rgba(15, 21, 35, 0.65);
        backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 20px;
        box-shadow: 0 16px 40px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s var(--spring-curve);
    }
    .apple-nav-header:hover {
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 48px -12px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 30%, #cbd5e1 70%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        margin: 0;
    }

    .brand-subtitle {
        color: #94a3b8;
        font-size: 0.88rem;
        font-weight: 400;
        letter-spacing: -0.01em;
        margin-top: 3px;
    }

    /* Doppelrand Double-Bezel Containers */
    .doppel-shell {
        background: var(--card-shell);
        border: 1px solid var(--card-shell-border);
        border-radius: 24px;
        padding: 8px;
        box-shadow: 0 12px 36px -8px rgba(0, 0, 0, 0.45);
        margin-bottom: 24px;
        transition: transform 0.35s var(--spring-curve), border-color 0.35s var(--spring-curve);
    }
    .doppel-inner {
        background: var(--card-inner);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 18px;
        padding: 24px;
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08);
    }

    /* Micro Eyebrow Tags */
    .eyebrow-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        background: rgba(255, 255, 255, 0.06);
        color: #cbd5e1;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 12px;
    }
    .eyebrow-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #38bdf8;
        box-shadow: 0 0 8px #38bdf8;
    }

    /* Verdict Card Styles with Apple-tier Depth */
    .verdict-outer {
        border-radius: 24px;
        padding: 8px;
        margin-bottom: 22px;
        box-shadow: 0 20px 48px -10px rgba(0, 0, 0, 0.55);
        transition: transform 0.4s var(--spring-curve);
    }
    .verdict-inner {
        border-radius: 18px;
        padding: 24px 28px;
        backdrop-filter: blur(20px);
        box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.12);
    }

    .v-LIKELY_MANIPULATED {
        background: radial-gradient(120% 120% at 10% 0%, rgba(244, 63, 94, 0.15) 0%, rgba(136, 19, 55, 0.06) 100%), rgba(22, 12, 18, 0.85);
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-top: 1px solid rgba(251, 113, 133, 0.55);
    }
    .v-LIKELY_MANIPULATED .verdict-title {
        color: #fda4af;
        text-shadow: 0 0 24px rgba(244, 63, 94, 0.45);
    }

    .v-SUSPICIOUS {
        background: radial-gradient(120% 120% at 10% 0%, rgba(245, 158, 11, 0.15) 0%, rgba(146, 64, 14, 0.06) 100%), rgba(24, 18, 12, 0.85);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-top: 1px solid rgba(251, 191, 36, 0.55);
    }
    .v-SUSPICIOUS .verdict-title {
        color: #fde68a;
        text-shadow: 0 0 24px rgba(245, 158, 11, 0.45);
    }

    .v-LIKELY_REAL {
        background: radial-gradient(120% 120% at 10% 0%, rgba(16, 185, 129, 0.15) 0%, rgba(6, 95, 70, 0.06) 100%), rgba(10, 22, 18, 0.85);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-top: 1px solid rgba(52, 211, 153, 0.55);
    }
    .v-LIKELY_REAL .verdict-title {
        color: #a7f3d0;
        text-shadow: 0 0 24px rgba(16, 185, 129, 0.45);
    }

    .v-UNCERTAIN {
        background: radial-gradient(120% 120% at 10% 0%, rgba(139, 92, 246, 0.15) 0%, rgba(76, 29, 149, 0.06) 100%), rgba(18, 14, 28, 0.85);
        border: 1px solid rgba(139, 92, 246, 0.35);
        border-top: 1px solid rgba(196, 181, 253, 0.55);
    }
    .v-UNCERTAIN .verdict-title {
        color: #ddd6fe;
        text-shadow: 0 0 24px rgba(139, 92, 246, 0.45);
    }

    .verdict-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin: 4px 0 12px 0;
    }

    /* Metric Badges & Pills */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }

    .confidence-meter {
        font-family: var(--font-mono);
        font-size: 0.95rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Forensic Basis Tray (Clean Native Styling) */
    .basis-tray {
        margin-top: 18px;
        padding: 16px 20px;
        background: rgba(8, 12, 22, 0.75);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.35);
    }
    .basis-heading {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #38bdf8;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }
    .basis-list {
        margin: 0;
        padding-left: 20px;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    .basis-item {
        margin-bottom: 6px;
    }
    .basis-item:last-child {
        margin-bottom: 0;
    }

    /* Detector Findings Cards */
    .detector-card {
        background: rgba(15, 21, 35, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: transform 0.2s var(--spring-curve), border-color 0.2s var(--spring-curve);
    }
    .detector-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.14);
        background: rgba(18, 25, 42, 0.75);
    }

    .status-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .status-SUCCESS { background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .status-UNCERTAIN { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .status-ERROR { background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.35); }
    .status-NOT_RUN { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); }

    /* Control Center Telemetry Tiles */
    .telemetry-tile {
        background: rgba(15, 21, 35, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }
    .telemetry-val {
        font-family: var(--font-mono);
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 6px 0 2px 0;
    }
    .telemetry-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Apple-Styled Primary Button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 9999px !important;
        padding: 14px 28px !important;
        font-size: 1.02rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        box-shadow: 0 8px 28px -4px rgba(37, 99, 235, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transition: all 0.25s var(--spring-curve) !important;
        cursor: pointer !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px) scale(1.01) !important;
        box-shadow: 0 12px 34px -4px rgba(37, 99, 235, 0.75), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
    }
    div.stButton > button:first-child:active {
        transform: scale(0.97) !important;
        transition: transform 80ms ease-out !important;
    }

    /* Streamlit Input Clean Glass Overrides */
    div[data-baseweb="textarea"] {
        background: rgba(10, 15, 26, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px !important;
        color: #f1f5f9 !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }
    div[data-baseweb="textarea"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
    }

    /* Custom File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(10, 15, 26, 0.55);
        border: 1px dashed rgba(255, 255, 255, 0.18);
        border-radius: 14px;
        padding: 12px;
        transition: all 0.25s var(--spring-curve);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #38bdf8;
        background: rgba(14, 21, 38, 0.65);
    }

    /* Streamlit Tab Segmented Control */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10, 14, 24, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 4px;
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #94a3b8;
        border: none;
        transition: all 0.2s var(--spring-curve);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
        background: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }

    /* Fact-Check Callout */
    .fc-callout {
        background: rgba(15, 23, 42, 0.65);
        border-left: 4px solid #38bdf8;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .fc-direct {
        border-left-color: #ef4444;
        background: rgba(30, 14, 20, 0.65);
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
# Floating Apple-Tier Navigation Header
# -----------------------------------------------------------------------------
render_html(
    """
    <div class="apple-nav-header">
        <div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <h1 class="brand-title">🛡️ VeriLens AI</h1>
                <span style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); padding: 3px 10px; border-radius: 9999px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">
                    FORENSIC INTELLIGENCE
                </span>
            </div>
            <p class="brand-subtitle">
                Multimodal Misinformation Detection & Autonomous Deepfake Forensic Analysis
            </p>
        </div>
        <div style="text-align: right; display: flex; align-items: center; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 9999px; padding: 5px 12px;">
                <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981;"></span>
                <span style="font-size: 0.78rem; font-weight: 700; color: #34d399; letter-spacing: 0.04em;">SYSTEM ONLINE</span>
            </div>
            <span style="font-family: var(--font-mono); font-size: 0.8rem; color: #64748b;">v1.0.0 Enterprise</span>
        </div>
    </div>
    """
)

# -----------------------------------------------------------------------------
# Sidebar: Apple System Settings Style (No API Key inputs)
# -----------------------------------------------------------------------------
# Auto-resolve API credentials from environment variables / .env in background
gemini_key = settings.GEMINI_API_KEY or None
fc_key = settings.FACT_CHECK_API_KEY or None

with st.sidebar:
    render_html(
        """
        <div style="padding: 4px 0 16px 0;">
            <div class="eyebrow-pill">
                <span class="eyebrow-dot"></span>
                <span>INSPECTOR SETTINGS</span>
            </div>
        </div>
        """
    )

    st.markdown("##### ⚙️ Execution Runtime")
    execution_mode = st.radio(
        "Execution Mode",
        ["Direct Orchestrator (In-Memory)", "FastAPI REST API (http://127.0.0.1:8000)"],
        index=0,
        label_visibility="collapsed",
    )
    api_url = st.text_input("FastAPI Base URL", value="http://127.0.0.1:8000")

    st.markdown("---")
    st.markdown("##### 🔒 Security & Credentials")
    
    # Elegant Apple-style credential vault indicator without exposing keys
    cred_status = "Authenticated & Active" if (gemini_key or fc_key) else "Local Forensic Isolation"
    cred_badge_bg = "rgba(16, 185, 129, 0.15)" if (gemini_key or fc_key) else "rgba(148, 163, 184, 0.15)"
    cred_badge_color = "#34d399" if (gemini_key or fc_key) else "#94a3b8"
    cred_badge_border = "rgba(16, 185, 129, 0.3)" if (gemini_key or fc_key) else "rgba(148, 163, 184, 0.25)"
    
    render_html(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 12px 14px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">Credentials Vault</span>
                <span style="background: {cred_badge_bg}; color: {cred_badge_color}; border: 1px solid {cred_badge_border}; border-radius: 9999px; padding: 2px 8px; font-size: 0.72rem; font-weight: 700;">
                    {cred_status}
                </span>
            </div>
            <p style="margin: 6px 0 0 0; font-size: 0.75rem; color: #64748b; line-height: 1.4;">
                API keys are sealed and managed via encrypted environment configurations.
            </p>
        </div>
        """
    )

    st.markdown("---")
    st.markdown("##### 🎯 Forensic Presets")
    st.caption("One-click benchmark scenarios for evaluation:")
    
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
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### 🧩 Active Forensic Channels")
    available_detectors = registry.list_all()
    enabled_detector_ids = []
    for d in available_detectors:
        checked = st.checkbox(f"{d.detector_name}", value=True, key=f"det_{d.detector_id}")
        if checked:
            enabled_detector_ids.append(d.detector_id)

    st.markdown("---")
    st.caption(f"Upload Ceiling: {settings.MAX_MEDIA_SIZE_BYTES // (1024*1024)} MB &bull; Max Text: {settings.MAX_TEXT_CHARS:,} chars")

# -----------------------------------------------------------------------------
# Preset Scenario Auto-Population
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Main Application Layout: Two-Column Apple Workspace
# -----------------------------------------------------------------------------
col_input, col_results = st.columns([1, 1.25], gap="large")

with col_input:
    render_html(
        """
        <div class="eyebrow-pill">
            <span class="eyebrow-dot"></span>
            <span>MULTIMODAL INGESTION</span>
        </div>
        """
    )
    
    input_text = st.text_area(
        "Headline / Claim / Caption Text",
        value=preset_text,
        height=140,
        placeholder="Enter headline text, viral tweet caption, or claim to cross-reference...",
        help="Input text will be cross-referenced with claim registries, verified encyclopedia sources, and examined for synthetic stylometry.",
    )
    
    uploaded_file = st.file_uploader(
        "Upload Media Asset (Image or Video)",
        type=["png", "jpg", "jpeg", "webp", "mp4", "mov"],
        help=f"Accepts JPG, PNG, WEBP, and MP4/MOV video clips up to {settings.MAX_MEDIA_SIZE_BYTES // (1024*1024)}MB",
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

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("Run Multimodal Verification →", type="primary", use_container_width=True)

with col_results:
    render_html(
        """
        <div class="eyebrow-pill">
            <span class="eyebrow-dot"></span>
            <span>INTELLIGENCE & FORENSIC DOSSIER</span>
        </div>
        """
    )
    
    if analyze_btn:
        if not input_text and not active_media_bytes:
            st.warning("Please supply at least one input modality (text or media file).")
        else:
            with st.spinner("Executing active forensic ensemble and cross-referencing knowledge graphs..."):
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
                    v_class = f"v-{v_key}"
                    
                    # Risk Badge Coloration
                    risk_color = (
                        "#f43f5e" if assessment.risk_level.value == "CRITICAL"
                        else "#fb7185" if assessment.risk_level.value == "HIGH"
                        else "#fbbf24" if assessment.risk_level.value == "MEDIUM"
                        else "#34d399"
                    )

                    # Build Verified Forensic Basis Items (No markdown code blocks)
                    basis_html = ""
                    if getattr(assessment, "basis_sources", None):
                        items_html = "".join([f"<li class='basis-item'>{s}</li>" for s in assessment.basis_sources])
                        basis_html = (
                            "<div class='basis-tray'>"
                            "<div class='basis-heading'><span>🔍</span><span>Verified Forensic Basis & Evidence Sources</span></div>"
                            f"<ul class='basis-list'>{items_html}</ul>"
                            "</div>"
                        )

                    # Render Apple-Tier Verdict Card (Pure HTML without indentation)
                    verdict_display = assessment.verdict.value.replace("_", " ")
                    card_html = (
                        f"<div class='verdict-outer {v_class}'>"
                        f"<div class='verdict-inner'>"
                        f"<div style='display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;'>"
                        f"<div>"
                        f"<div style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.14em; color: #94a3b8;'>OVERALL FORENSIC VERDICT</div>"
                        f"<h2 class='verdict-title'>{verdict_display}</h2>"
                        f"</div>"
                        f"<div style='text-align: right;'>"
                        f"<span class='metric-pill' style='color: {risk_color}; border-color: {risk_color}55;'>RISK: {assessment.risk_level.value}</span>"
                        f"<div style='margin-top: 6px;'><span class='confidence-meter'>Confidence: {int(assessment.confidence * 100)}%</span></div>"
                        f"</div>"
                        f"</div>"
                        f"<p style='margin: 0 0 12px 0; color: #e2e8f0; font-size: 0.96rem; line-height: 1.55;'>{assessment.summary}</p>"
                        f"{basis_html}"
                        f"</div>"
                        f"</div>"
                    )
                    render_html(card_html)

                    # Verified Fact-Check Banner (ClaimReview debunks)
                    verified_fcs = [e for e in response_data.evidence if e.evidence_type == "VERIFIED_FACT_CHECK"]
                    if verified_fcs:
                        direct_matches = [e for e in verified_fcs if (e.metadata or {}).get("is_direct_match", True)]
                        banner_title = "Direct Verified Fact-Check Debunk Found" if direct_matches else "Related Fact-Check Topic Context"
                        banner_border = "#ef4444" if direct_matches else "#0ea5e9"
                        banner_bg = "rgba(239, 68, 68, 0.12)" if direct_matches else "rgba(14, 165, 233, 0.12)"
                        banner_icon = "🚨" if direct_matches else "🌐"

                        fc_callouts_html = ""
                        for fc in verified_fcs:
                            meta = fc.metadata or {}
                            pub = meta.get("publisher", "Verified Fact-Checker")
                            rating = meta.get("rating", "Unrated")
                            url = meta.get("url", "")
                            claim = meta.get("claim", "")
                            is_direct = meta.get("is_direct_match", True)
                            date_val = meta.get("review_date", "")
                            date_str = f" &bull; <span style='color: #64748b;'>{date_val[:10]}</span>" if date_val else ""
                            link_markup = f"<a href='{url}' target='_blank' style='color: #38bdf8; text-decoration: underline; font-weight: 600; font-size: 0.85rem;'>Read Verified Source Article ↗</a>" if url else ""
                            
                            badge_bg = "rgba(239, 68, 68, 0.2)" if is_direct else "rgba(56, 189, 248, 0.15)"
                            badge_color = "#f87171" if is_direct else "#38bdf8"
                            badge_border = "#ef4444" if is_direct else "#0284c7"
                            badge_label = f"DIRECT MATCH &bull; RATING: {rating}" if is_direct else f"RELATED TOPIC &bull; RATING: {rating}"
                            note_markup = "" if is_direct else "<div style='color: #94a3b8; font-size: 0.78rem; font-style: italic; margin-top: 4px;'>ℹ️ Fact-checker evaluated a different rumor on this general topic, not your specific statement.</div>"

                            fc_callouts_html += (
                                f"<div class='fc-callout {'fc-direct' if is_direct else ''}'>"
                                f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                                f"<strong style='color: #f8fafc; font-size: 0.95rem;'>{pub}{date_str}</strong>"
                                f"<span style='background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border}; border-radius: 9999px; padding: 2px 10px; font-weight: 700; font-size: 0.78rem;'>{badge_label}</span>"
                                f"</div>"
                                f"<div style='color: #cbd5e1; font-size: 0.85rem; margin: 6px 0;'>Claim Reviewed: &ldquo;{claim}&rdquo;</div>"
                                f"<div>{link_markup}</div>"
                                f"{note_markup}"
                                f"</div>"
                            )

                        fc_banner_html = (
                            f"<div style='background: {banner_bg}; border: 1px solid {banner_border}; border-radius: 16px; padding: 18px 22px; margin-bottom: 22px;'>"
                            f"<div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>"
                            f"<span style='font-size: 1.25rem;'>{banner_icon}</span>"
                            f"<strong style='color: #f8fafc; font-size: 1.05rem;'>{banner_title} (ClaimReview Database)</strong>"
                            f"</div>"
                            f"<p style='color: #cbd5e1; font-size: 0.86rem; margin: 0 0 12px 0;'>"
                            f"{'This input claim directly corresponds to a verified debunk in ClaimReview archives:' if direct_matches else 'Independent fact-checkers investigated the following related rumors on this general topic:'}"
                            f"</p>"
                            f"{fc_callouts_html}"
                            f"</div>"
                        )
                        render_html(fc_banner_html)

                    # Primary Concerns
                    if assessment.primary_concerns:
                        with st.expander("⚠️ Primary Concerns Flagged", expanded=True):
                            for c in assessment.primary_concerns:
                                st.markdown(f"- {c}")

                    # Detailed Inspection Segmented Tabs
                    tab_findings, tab_evidence, tab_visualizer, tab_explanation, tab_limitations, tab_json = st.tabs(
                        ["🧩 Findings", "🔬 Evidence", "🖼️ Visualizer", "💡 Rationale", "⚠️ Governance", "🔌 JSON"]
                    )

                    with tab_findings:
                        for det_id, finding in response_data.findings.items():
                            status_badge = f"<span class='status-badge status-{finding.status.value}'>{finding.status.value}</span>"
                            score_text = (
                                f"{int(finding.manipulation_score * 100)}%"
                                if finding.manipulation_score is not None
                                else "N/A"
                            )
                            score_bar_width = (
                                f"{int(finding.manipulation_score * 100)}%"
                                if finding.manipulation_score is not None
                                else "0%"
                            )
                            score_bar_color = (
                                "#f43f5e" if (finding.manipulation_score or 0) >= 0.75
                                else "#fbbf24" if (finding.manipulation_score or 0) >= 0.50
                                else "#10b981"
                            )

                            det_card_html = (
                                f"<div class='detector-card'>"
                                f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>"
                                f"<strong style='font-size: 1.05rem; color: #f8fafc;'>{finding.detector_name}</strong>"
                                f"<div>{status_badge}</div>"
                                f"</div>"
                                f"<p style='margin: 4px 0 10px 0; color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;'>{finding.summary}</p>"
                                f"<div style='background: rgba(255,255,255,0.06); height: 5px; border-radius: 9999px; overflow: hidden; margin-bottom: 12px;'>"
                                f"<div style='background: {score_bar_color}; width: {score_bar_width}; height: 100%; border-radius: 9999px;'></div>"
                                f"</div>"
                                f"<div style='display: flex; gap: 24px; font-size: 0.82rem; color: #94a3b8;'>"
                                f"<span>Manipulation: <strong style='color: #38bdf8;'>{score_text}</strong></span>"
                                f"<span>Confidence: <strong style='color: #f1f5f9;'>{int(finding.confidence * 100)}%</strong></span>"
                                f"<span>Latency: <strong style='color: #f1f5f9;'>{finding.execution_time_ms} ms</strong></span>"
                                f"</div>"
                                f"</div>"
                            )
                            render_html(det_card_html)
                            if finding.metrics:
                                with st.expander(f"Telemetry metrics for {finding.detector_name}", expanded=False):
                                    st.json(finding.metrics)

                    with tab_evidence:
                        if response_data.evidence:
                            for ev in response_data.evidence:
                                sev_color = "#ef4444" if ev.severity.value == "HIGH" else "#f59e0b" if ev.severity.value == "MEDIUM" else "#10b981"
                                icon = "🌐" if ev.evidence_type == "VERIFIED_FACT_CHECK" else "🔬"
                                ev_card_html = (
                                    f"<div style='border-left: 4px solid {sev_color}; padding: 10px 18px; margin-bottom: 12px; background: rgba(15, 23, 42, 0.45); border-radius: 6px;'>"
                                    f"<div style='display: flex; justify-content: space-between; align-items: center;'>"
                                    f"<strong style='color: #f1f5f9; font-size: 0.92rem;'>{icon} [{ev.evidence_type}] {ev.description}</strong>"
                                    f"<span style='font-size: 0.75rem; color: {sev_color}; font-weight: 700; border: 1px solid {sev_color}44; padding: 2px 8px; border-radius: 9999px;'>{ev.severity.value}</span>"
                                    f"</div>"
                                    f"</div>"
                                )
                                render_html(ev_card_html)
                                if ev.metadata:
                                    if ev.evidence_type == "VERIFIED_FACT_CHECK" and ev.metadata.get("url"):
                                        st.markdown(f"🔗 **Fact-Check Reference:** [{ev.metadata.get('url')}]({ev.metadata.get('url')})")
                                    else:
                                        st.caption(f"Evidence Metadata: {json.dumps(ev.metadata)}")
                        else:
                            st.info("No concrete evidentiary anomalies extracted for this input.")

                    with tab_visualizer:
                        st.markdown("##### 🔬 Multi-Modal Forensic Visualizer (MVNN & DIDAN Insights)")
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
                                    st.markdown("###### 📷 Original Uploaded Media")
                                    st.image(orig, use_container_width=True)
                                with col_ela:
                                    st.markdown("###### 🔬 Error Level Analysis (ELA Heatmap)")
                                    st.image(ela_img, use_container_width=True)
                                    st.caption("Amplified pixel compression residuals (MVNN Pixel Domain). Spliced or in-painted regions stand out as high-intensity error clusters.")
                            except Exception as ex:
                                st.warning(f"Could not render ELA visualization: {ex}")
                        else:
                            st.info("Upload or select an image asset to inspect visual domain Error Level Analysis (ELA).")

                        # Forensics Telemetry Tiles (Apple Control-Center style)
                        syn_img_metrics = (response_data.findings.get("synthetic_image") or {}).metrics or {}
                        syn_txt_metrics = (response_data.findings.get("synthetic_text") or {}).metrics or {}
                        consist_metrics = (response_data.findings.get("consistency") or {}).metrics or {}
                        df_metrics = (response_data.findings.get("deepfake") or {}).metrics or {}

                        t1, t2, t3, t4 = st.columns(4)
                        with t1:
                            render_html(
                                f"<div class='telemetry-tile'><div class='telemetry-label'>2D FFT High-Freq</div><div class='telemetry-val'>{syn_img_metrics.get('high_frequency_ratio', 'N/A')}</div></div>"
                            )
                        with t2:
                            render_html(
                                f"<div class='telemetry-tile'><div class='telemetry-label'>ELA Residual</div><div class='telemetry-val'>{syn_img_metrics.get('ela_max_error', 'N/A')}</div></div>"
                            )
                        with t3:
                            render_html(
                                f"<div class='telemetry-tile'><div class='telemetry-label'>Warping / Jitter</div><div class='telemetry-val'>{df_metrics.get('morphological_warping', df_metrics.get('temporal_face_jitter', 'N/A'))}</div></div>"
                            )
                        with t4:
                            render_html(
                                f"<div class='telemetry-tile'><div class='telemetry-label'>DIDAN Delta</div><div class='telemetry-val'>{consist_metrics.get('didan_discrepancy_score', 'N/A')}</div></div>"
                            )

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
                        st.markdown("#### System Governance & Model Boundaries")
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
                            "📥 Download JSON Forensic Dossier",
                            data=json_str,
                            file_name=f"verilens_{response_data.request_id}.json",
                            mime="application/json",
                            use_container_width=True,
                        )
    else:
        st.info("Select a preset scenario from the sidebar or upload custom multimodal content to run verification.")
