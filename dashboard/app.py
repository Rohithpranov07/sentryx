"""
dashboard/app.py

SENTRY-X — Streamlit Demo Dashboard

Designed as a live demo surface for judges.
Connects to the FastAPI backend running on localhost:8000.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import requests
import json
import time
import io
import hashlib
from PIL import Image
from datetime import datetime
import base64

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SENTRY-X | Media Integrity Firewall",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }

  /* Dark background */
  .stApp {
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  /* Hide default Streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem; padding-bottom: 2rem; }

  /* ── Header bar ── */
  .sentry-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e40af33;
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .sentry-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.03em;
    margin: 0;
  }
  .sentry-subtitle {
    font-size: 0.8rem;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    margin: 0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    margin-right: 6px;
    animation: pulse-green 2s infinite;
  }
  @keyframes pulse-green {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 #22c55e66; }
    50% { opacity: 0.8; box-shadow: 0 0 0 6px #22c55e00; }
  }

  /* ── Risk verdict cards ── */
  .verdict-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid;
  }
  .verdict-green  { background: #052e16; border-color: #166534; }
  .verdict-yellow { background: #1c1400; border-color: #854d0e; }
  .verdict-orange { background: #1c0a00; border-color: #9a3412; }
  .verdict-red    { background: #1c0505; border-color: #991b1b; }

  .verdict-label {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
  }
  .verdict-green  .verdict-label { color: #4ade80; }
  .verdict-yellow .verdict-label { color: #facc15; }
  .verdict-orange .verdict-label { color: #fb923c; }
  .verdict-red    .verdict-label { color: #f87171; }

  .verdict-action {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-top: 0.4rem;
  }
  .action-publish  { background: #14532d; color: #4ade80; }
  .action-label    { background: #422006; color: #fbbf24; }
  .action-restrict { background: #431407; color: #fb923c; }
  .action-block    { background: #450a0a; color: #f87171; }

  /* ── Confidence bar ── */
  .conf-bar-wrapper {
    background: #1e293b;
    border-radius: 6px;
    height: 8px;
    margin: 0.8rem 0;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s ease;
  }

  /* ── Fingerprint display ── */
  .fp-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    word-break: break-all;
    line-height: 1.8;
  }
  .fp-label {
    color: #475569;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 2px;
  }
  .fp-value { color: #7dd3fc; }
  .fp-id    { color: #a78bfa; font-size: 0.9rem; font-weight: 700; }

  /* ── Forensic signals ── */
  .signal-item {
    background: #0f172a;
    border-left: 3px solid #334155;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.85rem;
    color: #cbd5e1;
    font-family: 'Space Mono', monospace;
  }

  /* ── Metric tiles ── */
  .metric-tile {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
  }
  .metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
    font-family: 'Space Mono', monospace;
  }

  /* ── Ledger badge ── */
  .ledger-badge {
    background: #1e1b4b;
    border: 1px solid #4338ca44;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #a5b4fc;
  }
  .ledger-badge .registered { color: #818cf8; font-weight: 700; }

  /* ── History table ── */
  .history-row {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 0.82rem;
  }
  .risk-badge {
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    white-space: nowrap;
  }
  .badge-green  { background: #14532d; color: #4ade80; }
  .badge-yellow { background: #422006; color: #fbbf24; }
  .badge-orange { background: #431407; color: #fb923c; }
  .badge-red    { background: #450a0a; color: #f87171; }

  /* ── Upload zone ── */
  .upload-hint {
    background: #0f172a;
    border: 2px dashed #1e40af44;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #475569;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #080c18 !important;
    border-right: 1px solid #1e293b;
  }

  /* Divider */
  hr { border-color: #1e293b !important; }

  /* Streamlit file uploader tweak */
  [data-testid="stFileUploader"] {
    background: #0f172a;
    border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "api_online" not in st.session_state:
    st.session_state.api_online = False
if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/v1/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def get_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/v1/health", timeout=3)
        return r.json()
    except Exception:
        return None


def analyze_image(file_bytes: bytes, filename: str, platform_id: str, caption: str = "") -> dict | None:
    try:
        files = {"file": (filename, io.BytesIO(file_bytes), "image/jpeg")}
        data  = {"platform_id": platform_id, "caption": caption, "uploader_id": "demo_admin"}
        r = requests.post(f"{API_BASE}/v2/analyze", files=files, data=data, timeout=60)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def risk_color(level: str) -> str:
    return {"green": "#22c55e", "yellow": "#eab308", "orange": "#f97316", "red": "#ef4444"}.get(level, "#94a3b8")


def risk_emoji(level: str) -> str:
    return {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(level, "⚪")


def action_badge_class(action: str) -> str:
    return f"action-{action}"


def conf_bar_color(confidence: float) -> str:
    if confidence < 0.30: return "#22c55e"
    if confidence < 0.55: return "#eab308"
    if confidence < 0.75: return "#f97316"
    return "#ef4444"


def format_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    if n < 1048576: return f"{n/1024:.1f} KB"
    return f"{n/1048576:.1f} MB"


# ── Header ────────────────────────────────────────────────────────────────────
st.session_state.api_online = check_api()
status_color = "#22c55e" if st.session_state.api_online else "#ef4444"
status_text  = "API ONLINE" if st.session_state.api_online else "API OFFLINE"

st.markdown(f"""
<div class="sentry-header">
  <div>
    <p class="sentry-title">🛡️ &nbsp;SENTRY-X</p>
    <p class="sentry-subtitle">Real-Time Media Integrity Firewall &nbsp;·&nbsp; AI4Dev '26 Hackathon</p>
  </div>
  <div style="text-align:right">
    <span style="font-family:'Space Mono',monospace; font-size:0.75rem; color:{status_color};">
      <span class="status-dot" style="background:{status_color};"></span>{status_text}
    </span><br>
    <span style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#475569;">
      V4.0 Proven Suite &nbsp;·&nbsp; ViT + Xception + GAN FFT
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.api_online:
    st.error("⚠️ Cannot reach SENTRY-X API at `localhost:8000`. Start the server first:\n```\nuvicorn app.main:app --reload --port 8000\n```")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    platform_id = st.text_input("Platform ID", value="ai4dev-demo", help="Identifier for the originating platform")
    st.markdown("---")
    
    st.markdown("### 🔑 API Keys")
    st.markdown("<div style='font-size:0.75rem; color:#94a3b8; margin-bottom:10px;'>Required for Online V4 Detector (Gemini, Midjourney detection)</div>", unsafe_allow_html=True)
    import os
    current_key = os.getenv("GEMINI_API_KEY", "")
    gemini_key = st.text_input("Gemini API Key", type="password", value=current_key)
    if gemini_key and gemini_key != current_key:
        import pathlib
        env_path = pathlib.Path(__file__).parent.parent / ".env"
        with open(env_path, "a") as f:
            f.write(f"\nGEMINI_API_KEY='{gemini_key}'\n")
        os.environ["GEMINI_API_KEY"] = gemini_key
        st.success("API Key saved!")
    
    st.markdown("---")

    # Live health panel
    st.markdown("### 📡 System Status")
    health = get_health()
    if health:
        st.markdown(f"""
        <div class="metric-tile" style="margin-bottom:0.5rem;">
          <div class="metric-value">{health['ledger_stats']['total_fingerprints']}</div>
          <div class="metric-label">Threats in Ledger</div>
        </div>
        """, unsafe_allow_html=True)

        by_risk = health["ledger_stats"].get("by_risk_level", {})
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-value" style="color:#ef4444;">{by_risk.get('red', 0)}</div>
              <div class="metric-label">🔴 Red</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-value" style="color:#f97316;">{by_risk.get('orange', 0)}</div>
              <div class="metric-label">🟠 Orange</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:0.5rem; font-family:'Space Mono',monospace; font-size:0.7rem; color:#475569;">
          Device: {health['model']['device']}<br>
          Model: {health['model']['name']}<br>
          Uptime: {int(health['uptime_seconds'])}s
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Session stats
    st.markdown("### 📊 This Session")
    total = len(st.session_state.history)
    if total > 0:
        risk_counts = {}
        for h in st.session_state.history:
            r = h.get("risk_level", "unknown")
            risk_counts[r] = risk_counts.get(r, 0) + 1

        for level in ["green", "yellow", "orange", "red"]:
            count = risk_counts.get(level, 0)
            if count > 0:
                st.markdown(f"{risk_emoji(level)} **{level.capitalize()}**: {count}")
    else:
        st.markdown("<span style='color:#475569;font-size:0.8rem;'>No analyses yet.</span>", unsafe_allow_html=True)

    if total > 0:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#334155; line-height:1.8;">
    SENTRY-X V4.0 Proven Pipeline<br>
    AI4Dev '26 · PSG College of Technology<br>
    Lead Dev: V. Rohith Pranov<br>
    Stack: FastAPI · ViT/XceptionNet/GAN-FFT
    </div>
    """, unsafe_allow_html=True)


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_analyze, tab_video, tab_history, tab_ledger, tab_about = st.tabs([
    "🖼️ Image Analysis",
    "🎬 Video Analysis",
    "📋 Session History",
    "🔗 Provenance Ledger",
    "ℹ️ About"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload Media")
        uploaded = st.file_uploader(
            "Drop an image file",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )

        if not uploaded:
            st.markdown("""
            <div class="upload-hint">
              📁 &nbsp;JPEG · PNG · WebP · BMP<br><br>
              Upload any image to analyze it for<br>
              deepfake manipulation and synthetic generation.<br><br>
              <strong style="color:#1e40af;">Tip for demo:</strong><br>
              Try a real photo → then a face from<br>
              thispersondoesnotexist.com
            </div>
            """, unsafe_allow_html=True)
        else:
            image = Image.open(uploaded)
            st.image(image, use_column_width=True)
            w, h = image.size
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#475569; margin-top:0.5rem;">
              {uploaded.name} &nbsp;·&nbsp; {w}×{h}px &nbsp;·&nbsp; {format_bytes(uploaded.size)}
            </div>
            """, unsafe_allow_html=True)
            
            caption = st.text_input("Context Caption", placeholder="Optional: Enter a caption to test Multilingual Intent Scanning")

        analyze_btn = st.button(
            "🔍 &nbsp; Analyze Now",
            disabled=not uploaded,
            use_container_width=True,
            type="primary",
        )

    with col_result:
        st.markdown("#### Analysis Result")

        if uploaded and analyze_btn:
            with st.spinner("Running deep multimodal analysis..."):
                file_bytes = uploaded.getvalue()
                caption_payload = caption if "caption" in locals() else ""
                t0 = time.time()
                result = analyze_image(file_bytes, uploaded.name, platform_id, caption_payload)
                elapsed_total = (time.time() - t0) * 1000

            if "error" in result:
                st.error(f"API error: {result['error']}")
            else:
                # V2 Parsing Mapping
                policy = result.get("amplification_policy", {})
                risk   = policy.get("tier", "green")
                action = policy.get("action", "publish")
                
                signals = result.get("detection_signals", {})
                conf = signals.get("fusion_threat_score", 1.0 if risk in ["red", "orange"] else 0)
                reason = policy.get("policy_enforcement", "Authentic")
                
                latency_profile = result.get("latency_profile_ms", {})
                v2_latency = latency_profile.get("total_pipeline_ms") or latency_profile.get("phase1_triage_ms", 0.0)
                
                
                generator = "Proven Ensemble" if signals else "Known Threat"

                # Store in history
                st.session_state.history.append({
                    "filename": uploaded.name,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "risk_level": risk,
                    "confidence": conf,
                    "action": action,
                    "processing_time_ms": v2_latency,
                    "fingerprint": result.get("fingerprints") or {},
                    "ledger": (result.get("threat_intelligence") or {}).get("ledger_txn")
                })

                # ── Verdict card ──
                verdict_label = {"green": "Authentic Media", "yellow": "AI-Generated Media", "orange": "Manipulated / Restricted", "red": "High-Risk Threat Blocked"}.get(risk, "Unknown")
                st.markdown(f"""
                <div class="verdict-card verdict-{risk}">
                  <div class="verdict-label">{risk_emoji(risk)} &nbsp;{verdict_label}</div>
                  <div>
                    <span class="verdict-action {action_badge_class(action)}">
                      {action.upper()}
                    </span>
                  </div>
                  <div style="margin-top:0.8rem; font-size:0.82rem; color:#94a3b8;">
                    {reason}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Confidence bar ──
                conf_pct  = int(conf * 100)
                bar_color = conf_bar_color(conf)
                st.markdown(f"""
                <div style="margin: 0.5rem 0;">
                  <div style="display:flex; justify-content:space-between; font-family:'Space Mono',monospace; font-size:0.72rem; color:#64748b; margin-bottom:4px;">
                    <span>MANIPULATION CONFIDENCE</span>
                    <span style="color:{bar_color}; font-weight:700;">{conf_pct}%</span>
                  </div>
                  <div class="conf-bar-wrapper">
                    <div class="conf-bar-fill" style="width:{conf_pct}%; background:{bar_color};"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Metrics row ──
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-tile">
                      <div class="metric-value" style="font-size:1.4rem;">{v2_latency:.0f}<span style="font-size:0.8rem;color:#64748b;">ms</span></div>
                      <div class="metric-label">V2 Latency</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-tile">
                      <div class="metric-value" style="font-size:1.4rem; color:{bar_color};">{conf:.3f}</div>
                      <div class="metric-label">Confidence</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-tile">
                      <div class="metric-value" style="font-size:0.95rem; line-height:1.3; color:#fbbf24; align-items:center; display:flex; justify-content:center; height:1.4rem;">{generator}</div>
                      <div class="metric-label">Detection Paradigm</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Forensic signals & NLP Intent ──
                st.markdown("**🔬 V2 Pipeline Audit Signals**")
                
                forensics = signals.get("forensic_signals", []) if signals else ["Triage: Payload directly intercepted by cross-platform ledger."]
                for sig in forensics:
                    st.markdown(f'<div class="signal-item">↳ {sig}</div>', unsafe_allow_html=True)
                    
                intent_sigs = result.get("intent_classification", {}).get("signals", [])
                for sig in intent_sigs:
                    st.markdown(f'<div class="signal-item" style="border-left-color:#8b5cf6;">🌐 {sig}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Fingerprint ──
                fp = result.get("fingerprint", {})
                st.markdown("**🔑 Media Fingerprint**")
                st.markdown(f"""
                <div class="fp-box">
                  <div class="fp-label">Fingerprint ID</div>
                  <div class="fp-id">{fp.get('fingerprint_id', '—')}</div>
                  <br>
                  <div class="fp-label">SHA-256 (cryptographic)</div>
                  <div class="fp-value">{fp.get('sha256', '—')}</div>
                  <br>
                  <div class="fp-label">pHash (perceptual)</div>
                  <div class="fp-value">{fp.get('phash', '—')}</div>
                  <br>
                  <div class="fp-label">dHash (adversarial-resistant)</div>
                  <div class="fp-value">{fp.get('dhash', '—')}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Ledger entry (only for red/orange) ──
                ledger = (result.get("threat_intelligence") or {}).get("ledger_txn")
                if ledger:
                    st.markdown("<br>", unsafe_allow_html=True)
                    ts = datetime.fromtimestamp(ledger.get("timestamp", 0)).strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"""
                    <div class="ledger-badge">
                      🔗 &nbsp;<span class="registered">REGISTERED TO PROVENANCE LEDGER</span><br>
                      ID: {ledger.get('fingerprint_id', '—')}<br>
                      Time: {ts}<br>
                      Node: {ledger.get('ledger_type', '—')}<br>
                      <span style="color:#4338ca; font-size:0.7rem;">{ledger.get('note', '')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                # ── PoC note ──
                with st.expander("ℹ️ PoC Model Note"):
                    st.markdown(f"""
                    <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#64748b; line-height:1.8;">
                    {result.get('poc_note', '')}
                    </div>
                    """, unsafe_allow_html=True)

        elif not uploaded:
            st.markdown("""
            <div style="height:300px; display:flex; align-items:center; justify-content:center; color:#334155; font-family:'Space Mono',monospace; font-size:0.8rem; text-align:center;">
              ← Upload an image to begin analysis
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    import sys
    import os
    if os.path.dirname(__file__) not in sys.path:
        sys.path.insert(0, os.path.dirname(__file__))
    from video_tab import render_video_tab
    render_video_tab()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.markdown("#### Session Analysis History")

    if not st.session_state.history:
        st.markdown("""
        <div style="color:#475569; font-family:'Space Mono',monospace; font-size:0.8rem; padding:2rem; text-align:center;">
          No analyses in this session yet.<br>Upload images in the Analyze tab to begin.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary bar
        total  = len(st.session_state.history)
        blocked = sum(1 for h in st.session_state.history if h.get("risk_level") == "red")
        flagged = sum(1 for h in st.session_state.history if h.get("risk_level") in ("orange", "yellow"))
        safe    = sum(1 for h in st.session_state.history if h.get("risk_level") == "green")

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label, color in [
            (c1, total, "TOTAL", "#94a3b8"),
            (c2, safe, "SAFE", "#22c55e"),
            (c3, flagged, "FLAGGED", "#f97316"),
            (c4, blocked, "BLOCKED", "#ef4444"),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-tile">
                  <div class="metric-value" style="color:{color};">{val}</div>
                  <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # History rows (newest first)
        for item in reversed(st.session_state.history):
            risk   = item.get("risk_level", "green")
            conf   = item.get("confidence", 0)
            action = item.get("action", "publish")
            ms     = item.get("processing_time_ms", 0)
            fp_id  = item.get("fingerprint", {}).get("fingerprint_id", "—")
            ts     = item.get("timestamp", "—")

            ledger_icon = "🔗" if item.get("ledger") else ""

            st.markdown(f"""
            <div class="history-row">
              <span class="risk-badge badge-{risk}">{risk_emoji(risk)} {risk.upper()}</span>
              <span style="color:#e2e8f0; flex:1; font-weight:600;">{item.get('filename','—')}</span>
              <span style="color:#64748b; font-family:'Space Mono',monospace; font-size:0.7rem;">conf: {conf:.3f}</span>
              <span style="color:#64748b; font-family:'Space Mono',monospace; font-size:0.7rem;">{ms:.0f}ms</span>
              <span style="color:#7dd3fc; font-family:'Space Mono',monospace; font-size:0.7rem;">{fp_id}</span>
              <span style="color:#64748b; font-family:'Space Mono',monospace; font-size:0.7rem;">{ts} {ledger_icon}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LEDGER LOOKUP
# ══════════════════════════════════════════════════════════════════════════════
with tab_ledger:
    st.markdown("#### Provenance Ledger Lookup")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#64748b; margin-bottom:1rem;">
    Query the permanent threat registry by SHA-256 fingerprint.<br>
    In production this queries a Polygon L2 smart contract. In this PoC, it queries the local SQLite ledger.
    </div>
    """, unsafe_allow_html=True)

    sha_input = st.text_input(
        "SHA-256 Fingerprint",
        placeholder="Paste a SHA-256 hash here (64 hex characters)...",
        label_visibility="collapsed",
    )

    # Quick fill from history
    if st.session_state.history:
        st.markdown("**Quick fill from session history:**")
        for item in st.session_state.history[-5:]:
            fp = item.get("fingerprint", {})
            sha = fp.get("sha256", "")
            risk = item.get("risk_level", "green")
            name = item.get("filename", "—")
            if sha:
                if st.button(f"{risk_emoji(risk)} {name}  —  {sha[:20]}...", key=f"fill_{sha[:8]}"):
                    sha_input = sha
                    st.rerun()

    if sha_input and len(sha_input) == 64:
        try:
            r = requests.get(f"{API_BASE}/v1/fingerprint/{sha_input}", timeout=5)
            data = r.json()

            if data.get("found"):
                ts_raw = data.get("first_seen")
                ts_fmt = datetime.fromtimestamp(ts_raw).strftime("%Y-%m-%d %H:%M:%S") if ts_raw else "—"
                risk = data.get("risk_level", "unknown")
                st.markdown(f"""
                <div class="verdict-card verdict-{risk}" style="margin-top:1rem;">
                  <div class="verdict-label">{risk_emoji(risk)} Known Threat Found</div>
                  <br>
                  <div class="fp-box">
                    <div class="fp-label">Fingerprint ID</div>
                    <div class="fp-id">{data.get('fingerprint_id', '—')}</div>
                    <br>
                    <div class="fp-label">SHA-256</div>
                    <div class="fp-value">{data.get('sha256','—')}</div>
                    <br>
                    <div class="fp-label">Risk Level</div>
                    <div class="fp-value">{risk.upper()}</div>
                    <br>
                    <div class="fp-label">Verdict</div>
                    <div class="fp-value">{data.get('verdict','—')}</div>
                    <br>
                    <div class="fp-label">First Registered</div>
                    <div class="fp-value">{ts_fmt}</div>
                    <br>
                    <div class="fp-label">Ledger Type</div>
                    <div class="fp-value">{data.get('ledger_type','—')}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-card verdict-green" style="margin-top:1rem;">
                  <div class="verdict-label">🟢 Not Found in Ledger</div>
                  <div style="margin-top:0.5rem; font-size:0.85rem; color:#94a3b8;">
                    SHA-256 <code style="color:#7dd3fc;">{sha_input[:20]}...</code> has no record in the threat registry.
                    This does not confirm authenticity — it means this exact file has not previously been flagged.
                  </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Ledger query failed: {e}")
    elif sha_input:
        st.warning("SHA-256 hash must be exactly 64 hex characters.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("#### SENTRY-X — System Architecture")

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.8rem; color:#94a3b8; line-height:2; background:#0f172a; padding:1.5rem; border-radius:10px; border:1px solid #1e293b;">
    User Upload<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#7dd3fc;">SENTRY-X Middleware API (FastAPI)</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#a78bfa;">Media Fingerprinting Engine</span>&nbsp;&nbsp;← SHA256 + pHash + dHash<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#f472b6;">Provenance Ledger Check</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;← SQLite (PoC) / Polygon L2 (Prod)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;↘<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;<span style="color:#22c55e;">FAST PATH: Known threat → instant block</span><br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#fb923c;">AI Forensic Engine</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;← EfficientNet-B4 (FF++ fine-tune pending)<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    <span style="color:#fbbf24;">Risk Classifier</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;← 🟢 🟡 🟠 🔴<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓<br>
    Platform Decision Layer&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;← Publish / Label / Restrict / Block<br>
    &nbsp;&nbsp;&nbsp;&nbsp;↓ (if RED or ORANGE)<br>
    <span style="color:#818cf8;">Ledger Write</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;← Permanent fingerprint registration
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Tech Stack**")
        for item in [
            ("🧠 AI Model", "EfficientNet-B4 (timm)"),
            ("⚡ API", "FastAPI + Uvicorn"),
            ("🔑 Fingerprinting", "SHA256 + pHash + dHash"),
            ("🔗 Ledger", "SQLite → Polygon L2"),
            ("🖥️ Dashboard", "Streamlit"),
            ("📦 Device", "MPS (Apple Silicon) / CUDA / CPU"),
        ]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-family:'Space Mono',monospace; font-size:0.75rem; padding:0.4rem 0; border-bottom:1px solid #1e293b; color:#94a3b8;">
              <span>{item[0]}</span><span style="color:#7dd3fc;">{item[1]}</span>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown("**PoC Honest Limitations**")
        for item in [
            "Model weights: ImageNet only (not FF++ fine-tuned)",
            "Image-only: video/audio is Phase 2",
            "Ledger: SQLite, not real blockchain",
            "Single instance: no Celery/Redis queue yet",
            "Forensic signals: proxy metrics, not GradCAM",
        ]:
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace; font-size:0.72rem; padding:0.4rem 0; border-bottom:1px solid #1e293b; color:#94a3b8;">
              ⚠ {item}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#334155; text-align:center;">
      SENTRY-X PoC · AI4Dev '26 · PSG College of Technology<br>
      Lead Developer: V. Rohith Pranov · MIT License
    </div>
    """, unsafe_allow_html=True)
