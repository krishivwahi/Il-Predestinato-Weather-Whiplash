"""Weather Whiplash: Live Track Condition Detector — Streamlit dashboard.

Main entry point for the Hugging Face Space. Wires together the vision engine
(CLIP zero-shot classification + CLS-token attention heatmaps), the MCDA
strategy pipeline, the temporal trend engine, and the simulated live-stream
loop into a single race-engineering dashboard.

Three input modes:
    1. Single Image Upload — classify one track frame instantly.
    2. Video Upload (MP4)  — sample and classify frames from a clip.
    3. Simulated Live Stream — wet → drying → dry transition, 1 frame / 1.5 s.
"""

from __future__ import annotations

import base64
import io
import math
import os
import tempfile
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

import trend_engine as te
from mcda_engine import ACTIONS, WEIGHTS, calculate_mcda_action
from simulation import TrackStreamSimulator
from vision_engine import CONDITION_LABELS, MODEL_ID, TrackVisionEngine

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Weather Whiplash — Live Track Condition Detector",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# F1 circuit catalogue — (latitude, longitude) for Open-Meteo
# ---------------------------------------------------------------------------
F1_CIRCUITS: Dict[str, Optional[Tuple[float, float]]] = {
    "Manual Input":               None,
    "Monaco (Monte Carlo)":       (43.7347,   7.4206),
    "Silverstone":                (52.0786,  -1.0169),
    "Monza":                      (45.6156,   9.2811),
    "Spa-Francorchamps":          (50.4372,   5.9714),
    "Singapore (Marina Bay)":     ( 1.2914, 103.8639),
    "Suzuka":                     (34.8431, 136.5407),
    "Interlagos (São Paulo)":     (-23.7036, -46.6997),
    "Zandvoort":                  (52.3888,   4.5409),
    "Baku (City Circuit)":        (40.3725,  49.8533),
    "Budapest (Hungaroring)":     (47.5789,  19.2486),
}

# ---------------------------------------------------------------------------
# CSS — dark racing theme with Inter font and animation keyframes
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {
    background: linear-gradient(170deg, #08090d 0%, #0d1117 60%, #111620 100%);
}
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p { color: #c9d1d9; }

h1 { color: #f0f6fc; letter-spacing: -0.5px; }
h2 { color: #e6edf3; }
h3 { color: #cdd9e5; font-size: 1.0rem; font-weight: 700;
     text-transform: uppercase; letter-spacing: 1px; }

div[data-testid="stMetricValue"]  { color: #e10600; font-weight: 800; }
div[data-testid="stMetricLabel"]  { color: #8b949e; font-size: 0.75rem;
                                     text-transform: uppercase; letter-spacing: 0.8px; }

/* General badge pill */
.ww-badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    font-weight: 700; font-size: 0.78rem; letter-spacing: 1.2px;
}

/* Metric cards row */
.ww-metric-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 10px 18px; flex: 1; min-width: 110px;
}
.ww-metric-card .val  { font-size: 1.25rem; font-weight: 800; color: #e10600; line-height: 1.1; }
.ww-metric-card .lbl  { font-size: 0.68rem; color: #8b949e; text-transform: uppercase;
                         letter-spacing: 0.9px; margin-top: 3px; }

/* Condition pills */
.ww-pill-row { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; margin: 10px 0; }
.ww-pill     { display: inline-block; padding: 3px 10px; border-radius: 999px;
                font-size: 0.72rem; font-weight: 700; letter-spacing: 0.8px;
                border: 1px solid currentColor; }
.ww-arrow    { color: #444c56; font-size: 1.1rem; line-height: 1; }

/* Crossover banner — pulsing orange glow */
@keyframes crossover-pulse {
  0%,100% { box-shadow: 0 0 0 0 rgba(198,74,0,0.50); }
  50%      { box-shadow: 0 0 0 14px rgba(198,74,0,0); }
}
.ww-crossover {
    background: linear-gradient(90deg, #7a2700, #c64a00);
    border-radius: 12px; padding: 14px 22px; margin: 12px 0;
    animation: crossover-pulse 2s ease-in-out infinite;
}
.ww-crossover .head { font-size: 1.05rem; font-weight: 800; color: #fff; }
.ww-crossover .sub  { color: rgba(255,255,255,0.88); font-size: 0.88rem; margin-top: 4px; }

/* Strategy card */
.ww-strategy-card { border-radius: 14px; padding: 18px 22px; margin-bottom: 10px; }
.ww-strategy-card .sub    { font-size: 0.70rem; letter-spacing: 2px;
                              color: rgba(255,255,255,0.55); text-transform: uppercase; }
.ww-strategy-card .action { font-size: 1.4rem; font-weight: 800; color: #fff; margin: 8px 0 4px 0; }
.ww-strategy-card .score  { color: rgba(255,255,255,0.85); font-weight: 600; font-size: 0.9rem; }

/* Footer */
.ww-footer { color: #6e7681; font-size: 0.82rem; line-height: 1.85; padding: 10px 0; }
.ww-footer a { color: #58a6ff; text-decoration: none; }
.ww-footer a:hover { text-decoration: underline; }

hr { border-color: #21262d !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------
CONDITION_COLORS: Dict[str, str] = {
    "Dry":    "#f5a623",
    "Damp":   "#7ed6df",
    "Wet":    "#2f6fed",
    "Drying": "#2ecc71",
}
ACTION_STYLES: Dict[str, Tuple[str, str]] = {
    "STAY_OUT":          ("#1e7d32", "🟢"),
    "PIT_SLICKS":        ("#c64a00", "🟠"),
    "PIT_INTERMEDIATES": ("#1256a8", "🔵"),
    "HOLD_EXTEND_STINT": ("#5b1298", "🟣"),
}
PIT_ACTIONS = frozenset({"PIT_SLICKS", "PIT_INTERMEDIATES"})


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="🔧 Warming up the CLIP vision engine…")
def get_engine() -> TrackVisionEngine:
    """Load TrackVisionEngine once per Space session."""
    return TrackVisionEngine()


@st.cache_data
def _synthesize_alert_wav() -> bytes:
    """Build a short pit-wall chirp as raw WAV bytes (no external assets)."""
    sr, dur = 22_050, 0.75
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    tone = (
        0.40 * np.sin(2 * math.pi * 880  * t) * np.exp(-3.2 * t)
        + 0.25 * np.sin(2 * math.pi * 1320 * t) * np.exp(-4.5 * t)
    )
    audio = (tone * 32_767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2)
        wav.setframerate(sr); wav.writeframes(audio.tobytes())
    return buf.getvalue()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_circuit_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Fetch live temperature and humidity from Open-Meteo (free, no API key).

    Results are cached for 5 minutes so rapid reruns don't hammer the API.

    Args:
        lat: Circuit latitude.
        lon: Circuit longitude.

    Returns:
        Dict with ``temperature_2m`` and ``relative_humidity_2m``, or ``None``
        if the network request fails.
    """
    try:
        import requests  # noqa: PLC0415
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m"
            "&temperature_unit=celsius"
        )
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        current = resp.json().get("current", {})
        return {
            "temperature_2m":       current.get("temperature_2m"),
            "relative_humidity_2m": current.get("relative_humidity_2m"),
        }
    except Exception:  # noqa: BLE001 — network error → graceful None
        return None


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_STATE_DEFAULTS: Dict[str, Any] = {
    "history":            [],    # list of classification result dicts
    "frame_count":        0,     # monotonic counter for chart widget key
    "last_alert_action":  None,  # deduplicates pit audio alerts
    "last_image_sig":     None,  # deduplicates single-image history pushes
    "track_temp_slider":  28,    # slider key — may be overwritten by weather fetch
    "humidity_slider":    55,    # slider key — may be overwritten by weather fetch
    "weather_fetch_msg":  "",    # status line shown below fetch button
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _push_history(frame_label: object, probs: Dict[str, float], action: str) -> None:
    """Append one classification record to the temporal trend buffer."""
    row: Dict[str, Any] = {"frame": str(frame_label), "action": action}
    row.update({label: round(probs[label], 4) for label in CONDITION_LABELS})
    st.session_state.history.append(row)
    st.session_state.frame_count += 1


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏎️ Weather Whiplash: Live Track Condition Detector")
st.markdown(
    f"<span class='ww-badge' style='background:#ffd21e22;color:#ffd21e;"
    f"border:1px solid #ffd21e44;'>"
    f"🤗 Hugging Face · {MODEL_ID}</span>",
    unsafe_allow_html=True,
)

engine = get_engine()
if engine.mock_mode:
    st.warning(
        "Vision model could not be loaded — running in **heuristic mock mode**. "
        "All dashboard features remain fully functional.",
        icon="⚠️",
    )
else:
    st.success(
        f"CLIP engine ready on `{engine.device}`. Upload a frame or start the stream.",
        icon="✅",
    )

# ---------------------------------------------------------------------------
# Sidebar — Strategy Control Panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ Strategy Control Panel")

    mode = st.radio(
        "Input Mode",
        ["Single Image Upload", "Video Upload (MP4)", "Simulated Live Stream"],
    )

    st.divider()
    st.subheader("Circuit Settings")
    circuit_type  = st.selectbox("Circuit Type", ["Standard Track", "Street Circuit"])
    is_street     = circuit_type == "Street Circuit"
    traffic_dense = st.toggle("Dense Traffic (midfield train)", value=False)

    # ── Weather & Environment ───────────────────────────────────────────────
    st.divider()
    st.subheader("🌦️ Weather & Environment")

    circuit_name   = st.selectbox(
        "F1 Circuit (auto-fill weather)",
        list(F1_CIRCUITS.keys()),
        help="Select a circuit to fetch live temperature and humidity from Open-Meteo.",
    )
    circuit_coords = F1_CIRCUITS.get(circuit_name)

    if circuit_coords and st.button("🌐 Fetch Live Weather", use_container_width=True):
        with st.spinner(f"Fetching weather for {circuit_name}…"):
            weather = _fetch_circuit_weather(*circuit_coords)
        if weather and weather["temperature_2m"] is not None:
            air_t   = float(weather["temperature_2m"])
            # Track surface is typically 10–20 °C above air temp in fine conditions;
            # use +14 °C as a conservative race-day estimate, clamped to slider range.
            track_t = max(10, min(50, int(air_t + 14)))
            hum     = max(0,  min(100, int(weather["relative_humidity_2m"] or 55)))
            st.session_state.track_temp_slider = track_t
            st.session_state.humidity_slider   = hum
            st.session_state.weather_fetch_msg = (
                f"✅ {circuit_name}: {air_t:.0f} °C air → ~{track_t} °C track est., "
                f"{hum}% RH"
            )
        else:
            st.session_state.weather_fetch_msg = (
                "❌ Could not reach Open-Meteo. Check your connection."
            )

    if st.session_state.weather_fetch_msg:
        st.caption(st.session_state.weather_fetch_msg)

    track_temp = st.slider(
        "Track Temperature (°C)", 10, 50,
        key="track_temp_slider",
        help="Auto-filled from Open-Meteo when a circuit is selected.",
    )
    humidity = st.slider(
        "Humidity (%)", 0, 100, step=5,
        key="humidity_slider",
        help="Ambient relative humidity. Higher values suppress track evaporation (w₄).",
    )

    # ── Telemetry Settings ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Telemetry Settings")
    lap_delta = st.slider("Lap Delta vs. baseline (s)", 0.0, 5.0, 2.5, 0.1)
    tire_wear = st.slider("Current Tyre Wear", 0.0, 1.0, 0.40, 0.05)

    # ── Vision ──────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Vision")
    show_heatmap = st.toggle(
        "Enable Grad-CAM Heatmap",
        value=False,
        help="Overlay CLS-token attention map highlighting wet/dry patches.",
    )

    if mode == "Video Upload (MP4)":
        sample_interval = st.slider("Frame sample interval (s)", 0.5, 5.0, 1.5, 0.5)
    if mode == "Simulated Live Stream":
        total_laps   = st.slider("Simulated laps", 5, 25, 15)
        start_stream = st.button("🚦 Start / Restart Stream", type="primary")
    else:
        start_stream = False
        total_laps   = 15

    st.divider()
    if st.button("🗑️ Reset Trend History", use_container_width=True):
        st.session_state.history          = []
        st.session_state.frame_count      = 0
        st.session_state.last_alert_action = None
        st.session_state.last_image_sig   = None
        st.rerun()


# ---------------------------------------------------------------------------
# Alert placeholder (above all modes)
# ---------------------------------------------------------------------------
alert_placeholder = st.empty()


# ===========================================================================
# UI Building Blocks
# ===========================================================================

def _render_metrics_row(
    probs: Dict[str, float],
    mcda: Dict[str, Any],
    velocity: float,
) -> None:
    """Five-card dashboard metrics row above the main vision panel.

    Cards: Top Condition · MCDA Confidence · Strategy Call · Drying Velocity · Frames
    """
    top_label  = max(probs, key=probs.get)  # type: ignore[arg-type]
    color      = CONDITION_COLORS[top_label]
    action_key = str(mcda["recommended_action"])
    _, icon    = ACTION_STYLES[action_key]
    v_label, v_icon, v_color = te.trend_label(velocity)
    v_pct      = f"{velocity:+.1%}/lap"

    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, color,   top_label.upper(),                          "Track Condition"),
        (c2, "#e10600", f"{mcda['confidence_score']:.0%}",        "MCDA Confidence"),
        (c3, ACTION_STYLES[action_key][0], f"{icon} {action_key.replace('_',' ')}", "Strategy Call"),
        (c4, v_color, f"{v_icon} {v_pct}",                        "Drying Velocity"),
        (c5, "#8b949e", str(st.session_state.frame_count),        "Frames Analysed"),
    ]
    for col, clr, val, lbl in cards:
        with col:
            st.markdown(
                f"<div class='ww-metric-card'>"
                f"<div class='val' style='color:{clr};font-size:1.05rem;'>{val}</div>"
                f"<div class='lbl'>{lbl}</div></div>",
                unsafe_allow_html=True,
            )


def _render_condition_pills() -> None:
    """Horizontal rolling-window pill sequence: Lap 1 ● WET → Lap 4 ● DAMP → …

    Shows the last 8 frames as colour-coded pills with arrow separators.
    """
    if not st.session_state.history:
        return
    sequence: List[Tuple[str, str]] = te.get_condition_sequence(
        st.session_state.history, n=8
    )
    parts: List[str] = []
    for i, (frame_lbl, cond) in enumerate(sequence):
        clr = CONDITION_COLORS[cond]
        parts.append(
            f"<span class='ww-pill' style='color:{clr};border-color:{clr}44;"
            f"background:{clr}18;'>{frame_lbl} ● {cond.upper()}</span>"
        )
        if i < len(sequence) - 1:
            parts.append("<span class='ww-arrow'>→</span>")

    st.markdown(
        f"<div class='ww-pill-row'>{''.join(parts)}</div>",
        unsafe_allow_html=True,
    )


def _render_crossover_banner(velocity: float, eta: Optional[float]) -> None:
    """Full-width pulsing orange banner when a wet→dry crossover window is active.

    Includes the drying velocity rate and an ETA estimate to the slick threshold.
    """
    eta_str = f" — slick window in **~{eta:.0f} laps**" if eta is not None else ""
    st.markdown(
        f"<div class='ww-crossover'>"
        f"<div class='head'>⚡ CROSSOVER WINDOW DETECTED</div>"
        f"<div class='sub'>"
        f"Dry-side probability accelerating at {velocity:+.1%}/lap{eta_str}. "
        f"Track trending dry — evaluate slick crossover now."
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_predictions(probs: Dict[str, float]) -> None:
    """Progress bars for the four condition probabilities + top-label badge."""
    top_label = max(probs, key=probs.get)  # type: ignore[arg-type]
    color     = CONDITION_COLORS[top_label]
    st.markdown(
        f"<span class='ww-badge' style='background:{color}22;color:{color};"
        f"border:1px solid {color}66;'>"
        f"TOP CALL: {top_label.upper()} — {probs[top_label]:.0%}</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    for label in CONDITION_LABELS:
        st.progress(min(1.0, float(probs[label])), text=f"{label}: {probs[label]:.1%}")


def _render_strategy_card(mcda: Dict[str, Any]) -> None:
    """High-visibility MCDA strategy call card with WSM breakdown expander."""
    action      = str(mcda["recommended_action"])
    color, icon = ACTION_STYLES[action]
    st.markdown(
        f"<div class='ww-strategy-card' style='background:{color};"
        f"box-shadow:0 6px 24px {color}55;'>"
        f"<div class='sub'>MCDA STRATEGY CALL · A*</div>"
        f"<div class='action'>{icon} {mcda['recommended_label']}</div>"
        f"<div class='score'>"
        f"Utility {max(mcda['scores_breakdown']['action_scores'].values()):.3f} · "
        f"Confidence {mcda['confidence_score']:.0%}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**📻 Pit wall:** {mcda['rationale']}")
    with st.expander("🔢 Weighted Sum Model breakdown"):
        breakdown = mcda["scores_breakdown"]
        st.dataframe(
            pd.DataFrame({
                "Action":   list(breakdown["action_scores"].keys()),
                "Score Sⱼ": [round(v, 4) for v in breakdown["action_scores"].values()],
            }).sort_values("Score Sⱼ", ascending=False),
            hide_index=True, use_container_width=True,
        )
        signals = breakdown["criteria_signals"]
        st.caption(
            "Signals: "
            + " · ".join(
                f"{k} = {v:.3f}" + (f" (w={WEIGHTS[k]:.2f})" if k in WEIGHTS else "")
                for k, v in signals.items()
            )
        )


def _render_trend_chart() -> None:
    """Real-time Plotly trend chart with per-condition velocity annotation.

    Uses ``session_state.frame_count`` as the widget key to guarantee
    re-render on every frame update without cross-mode key collisions.
    """
    st.subheader("📈 Temporal Condition Trend")
    if not st.session_state.history:
        st.info("No frames analysed yet — the trend chart fills as frames arrive.", icon="ℹ️")
        return

    df = pd.DataFrame(st.session_state.history)
    x  = list(range(1, len(df) + 1))

    fig = go.Figure()
    for label in CONDITION_LABELS:
        color = CONDITION_COLORS[label]
        y     = df[label].tolist()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=label,
            line=dict(color=color, width=3),
            marker=dict(size=6),
        ))

    # Annotate the drying velocity on the chart
    velocity = te.compute_drying_velocity(st.session_state.history)
    v_label, v_icon, _ = te.trend_label(velocity)
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.97,
        text=f"{v_icon} Drying velocity: {velocity:+.2%}/lap — {v_label}",
        showarrow=False,
        font=dict(size=12, color="#c9d1d9"),
        bgcolor="rgba(0,0,0,0.45)",
        borderpad=4,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,14,19,0.6)",
        height=360,
        margin=dict(l=8, r=8, t=8, b=8),
        xaxis_title="Frame / Lap",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(gridcolor="#21262d"),
    )
    st.plotly_chart(
        fig, use_container_width=True,
        key=f"trend_{st.session_state.frame_count}",
    )


def _render_analysis(
    image: Image.Image,
    probs: Dict[str, float],
    mcda: Dict[str, Any],
    show_hm: bool,
    eng: TrackVisionEngine,
    caption: str = "Input frame",
) -> None:
    """Full analysis panel: metrics row · pills · crossover banner · 3-col vision feed."""
    history = st.session_state.history
    velocity = te.compute_drying_velocity(history)
    crossover = te.detect_crossover_window(history)
    eta       = te.compute_eta_laps(history)

    # ── Metrics row ──────────────────────────────────────────────────────────
    _render_metrics_row(probs, mcda, velocity)

    # ── Rolling condition pills ───────────────────────────────────────────────
    _render_condition_pills()

    # ── Crossover banner (when window is active) ──────────────────────────────
    if crossover:
        _render_crossover_banner(velocity, eta)

    st.write("")
    # ── Three-column vision panel ────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.25, 1.0, 1.20])
    with col1:
        st.subheader("📷 Live Vision Feed")
        display_img = eng.generate_attention_heatmap(image) if show_hm else image
        st.image(
            display_img,
            caption=caption + (" · Grad-CAM overlay" if show_hm else ""),
            use_container_width=True,
        )
    with col2:
        st.subheader("🧠 Classification")
        _render_predictions(probs)
    with col3:
        st.subheader("🏁 Strategy Call")
        _render_strategy_card(mcda)


def _maybe_pit_alert(action: str, slot) -> None:
    """Play a pit-wall chirp when a high-priority pit window opens.

    Uses ``st.components.v1.html`` with a base64-encoded ``<audio autoplay>``
    tag — compatible with all Streamlit versions (autoplay kwarg removed ≥1.33).
    Deduplicates against ``last_alert_action`` to avoid repeat playback.
    """
    if action in PIT_ACTIONS:
        if st.session_state.last_alert_action != action:
            st.session_state.last_alert_action = action
            b64 = base64.b64encode(_synthesize_alert_wav()).decode()
            with slot.container():
                components.html(
                    f"<audio autoplay style='display:none;'>"
                    f"<source src='data:audio/wav;base64,{b64}' type='audio/wav'></audio>",
                    height=0,
                )
                st.toast(f"📻 PIT WINDOW OPEN — {ACTIONS[action]}", icon="🔔")
    else:
        st.session_state.last_alert_action = None


# ===========================================================================
# Mode: Single Image Upload
# ===========================================================================
if mode == "Single Image Upload":
    uploaded = st.file_uploader(
        "Upload a track surface frame",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )
    if uploaded is None:
        st.info("👆 Upload a trackside or onboard frame to run the detector.", icon="📂")
    else:
        image = Image.open(uploaded).convert("RGB")
        probs = engine.classify_track(image)
        mcda  = calculate_mcda_action(
            probs, lap_delta, is_street, traffic_dense,
            float(track_temp), tire_wear, float(humidity),
        )
        sig = (uploaded.name, uploaded.size, lap_delta, track_temp, tire_wear,
               humidity, is_street, traffic_dense)
        if st.session_state.last_image_sig != sig:
            st.session_state.last_image_sig = sig
            _push_history(uploaded.name, probs, str(mcda["recommended_action"]))
        _render_analysis(image, probs, mcda, show_heatmap, engine, caption=uploaded.name)
        _maybe_pit_alert(str(mcda["recommended_action"]), alert_placeholder)

    st.divider()
    _render_trend_chart()

# ===========================================================================
# Mode: Video Upload (MP4)
# ===========================================================================
elif mode == "Video Upload (MP4)":
    video_file = st.file_uploader(
        "Upload onboard / trackside video",
        type=["mp4", "mov", "avi"],
    )
    if video_file is None:
        st.info("👆 Upload a short clip — frames are sampled and classified.", icon="🎬")
    elif st.button("🎬 Analyse Video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        try:
            with st.spinner("Sampling and classifying frames…"):
                samples = engine.process_video_file(tmp_path, sample_interval_sec=sample_interval)
        finally:
            os.unlink(tmp_path)

        if not samples:
            st.error("No frames could be extracted from this file.", icon="❌")
        else:
            st.session_state.history     = []
            st.session_state.frame_count = 0
            last: Optional[tuple] = None

            for timestamp, frame, probs in samples:
                mcda = calculate_mcda_action(
                    probs, lap_delta, is_street, traffic_dense,
                    float(track_temp), tire_wear, float(humidity),
                )
                _push_history(f"t={timestamp}s", probs, str(mcda["recommended_action"]))
                last = (timestamp, frame, probs, mcda)

            assert last is not None
            timestamp, frame, probs, mcda = last
            st.success(f"Analysed **{len(samples)} frames** (last at t = {timestamp} s).", icon="✅")
            _render_analysis(
                frame, probs, mcda, show_heatmap, engine,
                caption=f"Last sampled frame (t = {timestamp} s)",
            )
            _maybe_pit_alert(str(mcda["recommended_action"]), alert_placeholder)

    st.divider()
    _render_trend_chart()

# ===========================================================================
# Mode: Simulated Live Stream
# ===========================================================================
else:
    st.caption(
        "Simulates a wet → drying → dry race transition, one frame every 1.5 s, "
        "with telemetry and humidity evolving lap by lap."
    )
    top_placeholder   = st.empty()
    chart_placeholder = st.empty()

    if start_stream:
        st.session_state.history           = []
        st.session_state.frame_count       = 0
        st.session_state.last_alert_action = None
        simulator = TrackStreamSimulator(total_frames=total_laps)

        for frame_id, image, telemetry in simulator.stream():
            probs = engine.classify_track(image)
            mcda  = calculate_mcda_action(
                probs,
                telemetry["lap_delta"],
                is_street,
                traffic_dense,
                telemetry["track_temp_c"],
                telemetry["tire_wear_pct"],
                telemetry["humidity_pct"],          # ← live simulated humidity
            )
            _push_history(f"Lap {frame_id}", probs, str(mcda["recommended_action"]))

            with top_placeholder.container():
                _render_analysis(
                    image, probs, mcda, show_heatmap, engine,
                    caption=(
                        f"Lap {telemetry['lap']} · Δ {telemetry['lap_delta']:+.2f} s · "
                        f"{telemetry['track_temp_c']:.0f} °C · "
                        f"{telemetry['humidity_pct']:.0f}% RH · "
                        f"wear {telemetry['tire_wear_pct']:.0%}"
                    ),
                )
            with chart_placeholder.container():
                _render_trend_chart()
            _maybe_pit_alert(str(mcda["recommended_action"]), alert_placeholder)
            time.sleep(1.5)

        st.success("🏁 Stream complete — full wet → dry transition analysed!")
    else:
        with chart_placeholder.container():
            _render_trend_chart()


# ---------------------------------------------------------------------------
# Footer — HF compliance
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
<div class="ww-footer">
<b>🤗 Hugging Face Hub Compliance</b><br>
Vision model: <a href="https://huggingface.co/openai/clip-vit-base-patch32" target="_blank">openai/clip-vit-base-patch32</a>
(zero-shot CLIP) ·
Alt: <a href="https://huggingface.co/google/vit-base-patch16-224" target="_blank">google/vit-base-patch16-224</a><br>
Dataset: publish via <code>dataset_setup.py</code> →
<a href="https://huggingface.co/datasets" target="_blank">HF Dataset Hub</a><br>
Weather data: <a href="https://open-meteo.com/" target="_blank">Open-Meteo API</a> (free, no key required)<br>
Team:
<a href="https://huggingface.co/" target="_blank">HF/&lt;member-1&gt;</a> ·
<a href="https://huggingface.co/" target="_blank">HF/&lt;member-2&gt;</a> ·
<a href="https://huggingface.co/" target="_blank">HF/&lt;member-3&gt;</a><br>
Built with Streamlit · Transformers · Plotly · OpenCV — <i>Weather Whiplash, F1 Hackathon 2026</i>.
</div>
""",
    unsafe_allow_html=True,
)
