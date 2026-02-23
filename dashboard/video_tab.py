"""
dashboard/video_tab.py

SENTRY-X Video Analysis Tab for Streamlit Dashboard

This module renders the entire "🎬 Video Analysis" tab.
Imported and called from dashboard/app.py.

Features:
  - Video upload with format validation
  - Configurable frame sampling rate
  - Real-time frame processing progress bar
  - Timeline chart showing per-frame confidence
  - Peak frame display (worst detected frame)
  - Temporal analysis forensic note
  - Full verdict card with flagged timestamps
  - Ledger registration display
"""

import streamlit as st
import requests
import io
import time
from datetime import datetime
from PIL import Image


API_BASE = "http://localhost:8000"
SUPPORTED_EXTENSIONS = ["mp4", "mov", "avi", "webm", "mkv"]


def risk_emoji(level: str) -> str:
    return {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}.get(level, "⚪")


def risk_color(level: str) -> str:
    return {
        "green":  "#22c55e",
        "yellow": "#eab308",
        "orange": "#f97316",
        "red":    "#ef4444",
    }.get(level, "#94a3b8")


def format_bytes(n: int) -> str:
    if n < 1048576: return f"{n/1024:.1f} KB"
    return f"{n/1048576:.1f} MB"


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m > 0 else f"{s}s"


def render_timeline_chart(timeline_data: list):
    """
    Render a styled confidence-over-time chart using Streamlit native charts.
    Colors each bar by risk level.
    """
    if not timeline_data:
        return

    # Color map for risk
    color_map = {
        "green":  "#22c55e",
        "yellow": "#eab308",
        "orange": "#f97316",
        "red":    "#ef4444",
    }

    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#64748b; margin-bottom:0.5rem;">
      FRAME-BY-FRAME MANIPULATION CONFIDENCE TIMELINE
    </div>
    """, unsafe_allow_html=True)

    # Build HTML bar chart (more control than st.bar_chart for colors)
    max_conf = max(d["confidence"] for d in timeline_data) if timeline_data else 100
    chart_html = '<div style="display:flex; align-items:flex-end; gap:3px; height:120px; background:#0f172a; padding:10px; border-radius:8px; border:1px solid #1e293b; overflow-x:auto;">'

    for d in timeline_data:
        pct   = d["confidence"]
        color = color_map.get(d["risk_level"], "#94a3b8")
        bar_h = max(4, int((pct / 100) * 100)) if max_conf > 0 else 4
        ts    = d["timestamp"]

        chart_html += f"""
        <div style="display:flex; flex-direction:column; align-items:center; min-width:22px;" title="t={ts:.1f}s | {pct:.1f}%">
          <div style="flex:1; display:flex; align-items:flex-end;">
            <div style="width:14px; height:{bar_h}px; background:{color}; border-radius:2px 2px 0 0;"></div>
          </div>
          <div style="font-size:9px; color:#475569; margin-top:2px; writing-mode:vertical-lr; transform:rotate(180deg); height:24px; overflow:hidden;">{ts:.0f}s</div>
        </div>"""

    chart_html += '</div>'

    # Risk level legend
    chart_html += """
    <div style="display:flex; gap:1rem; margin-top:0.5rem; font-family:'Space Mono',monospace; font-size:0.65rem;">
      <span style="color:#22c55e;">■ Authentic</span>
      <span style="color:#eab308;">■ AI-Generated</span>
      <span style="color:#f97316;">■ Suspicious</span>
      <span style="color:#ef4444;">■ Manipulated</span>
    </div>"""

    st.markdown(chart_html, unsafe_allow_html=True)


def render_video_tab():
    """Main entry point — renders the full Video Analysis tab."""

    st.markdown("#### 🎬 Video Analysis")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#64748b; margin-bottom:1.2rem; line-height:1.8;">
    Upload a video to run frame-by-frame deepfake detection.<br>
    SENTRY-X extracts frames, analyzes each one independently,<br>
    then aggregates results to find exactly <em>where</em> manipulation occurs.
    </div>
    """, unsafe_allow_html=True)

    # ── Upload + Config row ───────────────────────────────────────────────────
    col_up, col_cfg = st.columns([2, 1], gap="large")

    with col_up:
        uploaded_video = st.file_uploader(
            "Upload video",
            type=SUPPORTED_EXTENSIONS,
            label_visibility="collapsed",
            help="MP4, MOV, AVI, WebM, MKV — max 200MB",
        )

        if not uploaded_video:
            st.markdown("""
            <div style="background:#0f172a; border:2px dashed #1e40af44; border-radius:12px; padding:2rem; text-align:center; color:#475569; font-family:'Space Mono',monospace; font-size:0.78rem; line-height:2;">
              📹 &nbsp;MP4 · MOV · AVI · WebM · MKV<br>
              Max 200MB &nbsp;·&nbsp; Up to 60 frames analyzed<br><br>
              <strong style="color:#3b82f644;">Demo tip:</strong><br>
              Download a short clip from YouTube<br>
              or use any video from your device
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#64748b; padding:0.5rem 0;">
              📹 &nbsp;<span style="color:#e2e8f0;">{uploaded_video.name}</span>
              &nbsp;·&nbsp; {format_bytes(uploaded_video.size)}
            </div>
            """, unsafe_allow_html=True)

    with col_cfg:
        st.markdown("**Analysis Settings**")

        sample_fps = st.slider(
            "Frames per second to sample",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.5,
            help=(
                "1.0 = 1 frame per second (default, fast)\n"
                "2.0 = 2 frames per second (more thorough)\n"
                "3.0 = 3 frames per second (slowest, most accurate)"
            ),
        )

        platform_id = st.text_input(
            "Platform ID",
            value="ai4dev-demo",
            label_visibility="visible",
        )

        if uploaded_video:
            est_size_mb = uploaded_video.size / 1048576
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#475569; line-height:1.8;">
              Size: {est_size_mb:.1f}MB<br>
              Sample rate: {sample_fps} fps<br>
              Max frames: 60
            </div>
            """, unsafe_allow_html=True)

    # ── Analyze button ────────────────────────────────────────────────────────
    analyze_btn = st.button(
        "🎬 &nbsp; Analyze Video",
        disabled=not uploaded_video,
        use_container_width=True,
        type="primary",
    )

    # ── Analysis + Results ────────────────────────────────────────────────────
    if uploaded_video and analyze_btn:
        file_bytes = uploaded_video.read()

        # Progress UI
        progress_bar  = st.progress(0, text="Uploading video to SENTRY-X API...")
        status_text   = st.empty()

        t_start = time.time()

        try:
            progress_bar.progress(10, text="Extracting frames...")

            files = {"file": (uploaded_video.name, io.BytesIO(file_bytes), "video/mp4")}
            data  = {"platform_id": platform_id, "sample_fps": sample_fps}

            # Stream the request (can take a while for longer videos)
            status_text.markdown(
                '<span style="font-family:Space Mono,monospace; font-size:0.75rem; color:#64748b;">'
                f'Analyzing at {sample_fps} fps — this may take 30–120s depending on video length and device...</span>',
                unsafe_allow_html=True
            )
            progress_bar.progress(25, text="Running AI forensic detection on frames...")

            response = requests.post(
                f"{API_BASE}/v1/analyze/video",
                files=files,
                data=data,
                timeout=300,  # 5 min timeout for longer videos
            )

            progress_bar.progress(90, text="Aggregating results...")

            if response.status_code != 200:
                progress_bar.empty()
                status_text.empty()
                st.error(f"API error {response.status_code}: {response.text[:500]}")
                return

            result = response.json()
            progress_bar.progress(100, text="Done.")
            elapsed_total = time.time() - t_start

            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()
            
            # Store in history
            v_info = result.get("verdict", {})
            st.session_state.history.append({
                "risk_level": v_info.get("risk_level", "green"),
                "verdict": v_info.get("verdict", "Authentic"),
                "action": v_info.get("action", "publish"),
                "confidence": v_info.get("max_confidence", 0),
                "processing_time_ms": v_info.get("processing_time_ms", 0),
                "fingerprint": {
                    "fingerprint_id": result.get("fingerprint_id", "—"),
                    "sha256": result.get("sha256", "")
                },
                "ledger": result.get("ledger_registered", False),
                "filename": uploaded_video.name,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "type": "video"
            })

        except requests.exceptions.Timeout:
            progress_bar.empty()
            status_text.empty()
            st.error("Request timed out. Try a shorter video or reduce sample_fps to 0.5.")
            return
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Analysis failed: {e}")
            return

        # ── Render results ────────────────────────────────────────────────────
        _render_video_results(result, elapsed_total)


def _render_video_results(result: dict, elapsed_total: float):
    """Render the full results panel for a completed video analysis."""

    verdict   = result.get("verdict", {})
    risk      = verdict.get("risk_level", "green")
    frames    = result.get("frame_results", [])
    timeline  = result.get("timeline_data", [])
    meta      = result.get("video_metadata", {})
    counts    = verdict.get("frame_counts", {})

    st.markdown("---")
    st.markdown("### Results")

    # ── Top verdict card ──────────────────────────────────────────────────────
    action_class = {
        "publish":  "action-publish",
        "label":    "action-label",
        "restrict": "action-restrict",
        "block":    "action-block",
    }.get(verdict.get("action", "publish"), "action-publish")

    st.markdown(f"""
    <div class="verdict-card verdict-{risk}">
      <div class="verdict-label">{risk_emoji(risk)} &nbsp;{verdict.get('verdict','')}</div>
      <span class="verdict-action {action_class}">{verdict.get('action','').upper()}</span>
      <div style="margin-top:0.8rem; font-size:0.85rem; color:#94a3b8;">
        {verdict.get('description', '')}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Video metadata row ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)

    for col, val, label, color in [
        (m1, format_duration(meta.get("duration_seconds", 0)), "DURATION",   "#94a3b8"),
        (m2, meta.get("resolution", "—"),                      "RESOLUTION", "#94a3b8"),
        (m3, verdict.get("total_frames_analyzed", 0),          "FRAMES",     "#7dd3fc"),
        (m4, f"{verdict.get('max_confidence', 0)*100:.1f}%",   "PEAK CONF",  risk_color(risk)),
        (m5, f"{int(verdict.get('manipulation_ratio',0)*100)}%","FLAGGED",   risk_color(risk)),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-value" style="font-size:1.2rem; color:{color};">{val}</div>
              <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── Frame counts ──────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    total_f = verdict.get("total_frames_analyzed", 1)
    fc1, fc2, fc3, fc4 = st.columns(4)
    for col, level, label in [
        (fc1, "green",  "🟢 Authentic"),
        (fc2, "yellow", "🟡 AI-Gen"),
        (fc3, "orange", "🟠 Suspicious"),
        (fc4, "red",    "🔴 Manipulated"),
    ]:
        count = counts.get(level, 0)
        pct   = int(count / total_f * 100) if total_f > 0 else 0
        with col:
            st.markdown(f"""
            <div class="metric-tile">
              <div class="metric-value" style="font-size:1.3rem; color:{risk_color(level)};">{count}</div>
              <div class="metric-label">{label} ({pct}%)</div>
            </div>""", unsafe_allow_html=True)

    # ── Timeline chart ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    render_timeline_chart(timeline)

    # ── Temporal forensic note ────────────────────────────────────────────────
    consistency   = verdict.get("temporal_consistency", "")
    temporal_note = verdict.get("temporal_note", "")
    consistency_color = {
        "consistent":   "#22c55e",
        "burst":        "#ef4444",
        "inconsistent": "#f97316",
    }.get(consistency, "#94a3b8")

    st.markdown(f"""
    <div style="background:#0f172a; border:1px solid #1e293b; border-left:3px solid {consistency_color}; border-radius:0 8px 8px 0; padding:1rem; margin:1rem 0; font-family:'Space Mono',monospace; font-size:0.78rem; color:#94a3b8; line-height:1.7;">
      <span style="color:{consistency_color}; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">
        ⏱ Temporal Pattern: {consistency.upper()}
      </span><br><br>
      {temporal_note}
    </div>
    """, unsafe_allow_html=True)

    # ── Flagged timestamps ────────────────────────────────────────────────────
    flagged_ts = verdict.get("flagged_timestamps", [])
    if flagged_ts:
        ts_list = " &nbsp;·&nbsp; ".join([f"<span style='color:#f87171;'>{t:.1f}s</span>" for t in flagged_ts])
        st.markdown(f"""
        <div style="background:#1c0505; border:1px solid #7f1d1d; border-radius:8px; padding:0.8rem 1rem; margin:0.5rem 0; font-family:'Space Mono',monospace; font-size:0.78rem;">
          <span style="color:#f87171; font-weight:700;">🚨 FLAGGED TIMESTAMPS:</span><br>
          {ts_list}
        </div>
        """, unsafe_allow_html=True)

    # ── Peak frame detail ─────────────────────────────────────────────────────
    peak_ts  = verdict.get("peak_timestamp_s", 0)
    peak_idx = verdict.get("peak_frame_index", 0)
    peak_frame_data = next(
        (f for f in frames if f["frame_index"] == peak_idx), None
    )

    if peak_frame_data:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**🔍 Peak Manipulation Frame**")

        pf_col1, pf_col2 = st.columns([1, 2])
        with pf_col1:
            st.markdown(f"""
            <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:1rem; font-family:'Space Mono',monospace; font-size:0.75rem; color:#94a3b8; line-height:2;">
              <div style="color:{risk_color(peak_frame_data['risk_level'])}; font-size:1.1rem; font-weight:700; margin-bottom:0.5rem;">
                {risk_emoji(peak_frame_data['risk_level'])} {peak_frame_data['verdict']}
              </div>
              Timestamp: {peak_frame_data['timestamp_s']:.2f}s<br>
              Frame: #{peak_frame_data['frame_index']}<br>
              Confidence: {peak_frame_data['confidence']*100:.1f}%<br>
              Action: {peak_frame_data['action'].upper()}
            </div>
            """, unsafe_allow_html=True)

        with pf_col2:
            st.markdown("**Forensic Signals at Peak Frame:**")
            for sig in peak_frame_data.get("forensic_signals", []):
                st.markdown(
                    f'<div class="signal-item">↳ {sig}</div>',
                    unsafe_allow_html=True
                )

    # ── All frame results (expandable) ───────────────────────────────────────
    with st.expander(f"📋 All Frame Results ({len(frames)} frames)"):
        for fr in frames:
            level = fr["risk_level"]
            conf  = fr["confidence"]
            ts    = fr["timestamp_s"]
            bar_w = int(conf * 100)
            bar_c = risk_color(level)

            st.markdown(f"""
            <div class="history-row" style="margin:0.25rem 0;">
              <span class="risk-badge badge-{level}">{risk_emoji(level)} {level.upper()}</span>
              <span style="color:#64748b; font-family:'Space Mono',monospace; font-size:0.7rem; min-width:40px;">t={ts:.1f}s</span>
              <div style="flex:1; background:#1e293b; border-radius:4px; height:6px; overflow:hidden;">
                <div style="width:{bar_w}%; height:100%; background:{bar_c}; border-radius:4px;"></div>
              </div>
              <span style="color:{bar_c}; font-family:'Space Mono',monospace; font-size:0.7rem; min-width:40px;">{conf*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    # ── Ledger registration ───────────────────────────────────────────────────
    if result.get("ledger_registered"):
        st.markdown(f"""
        <div class="ledger-badge" style="margin-top:1rem;">
          🔗 &nbsp;<span class="registered">REGISTERED TO PROVENANCE LEDGER</span><br>
          Video ID: {result.get('fingerprint_id', '—')}<br>
          SHA-256: {result.get('sha256','—')[:32]}...<br>
          <span style="color:#4338ca; font-size:0.7rem;">{result.get('ledger_note','')}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── PoC note ──────────────────────────────────────────────────────────────
    with st.expander("ℹ️ PoC Model Note"):
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#64748b; line-height:1.8;">
        {result.get('poc_note', '')}<br><br>
        Processing time: {result['verdict'].get('processing_time_ms', 0):.0f}ms total<br>
        Device: {result.get('device_used', '—')}<br>
        Model: {result.get('model', '—')}
        </div>
        """, unsafe_allow_html=True)
