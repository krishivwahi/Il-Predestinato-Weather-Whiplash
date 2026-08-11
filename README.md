---
title: Weather Whiplash — Live Track Condition Detector
---

# Weather Whiplash: Live Track Condition Detector

An AI race-strategy dashboard that watches the track surface in real time,
classifies it as **Dry, Damp, Wet, or Drying** using zero-shot CLIP from the
Hugging Face Hub, tracks the temporal probability trend lap by lap, and
converts vision + telemetry + track context into a concrete pit-wall call
through a **Multi-Criteria Decision Analysis (MCDA)** engine.

> Track conditions change faster than weather reports. On a street circuit,
> pitting one lap too early or too late costs track position you never recover.
> **Weather Whiplash** answers the only question that matters on the pit wall:
> *is the track getting better or worse — and what do we do about it right now?*

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  INGESTION & SIMULATION LAYER               │
│  simulation.py                                              │
│  • Single image upload   • MP4 video upload (OpenCV)        │
│  • Simulated live stream (1 frame / 1.5 s, wet→dry)         │
│  • Telemetry: circuit type, traffic, lap Δ, temp, wear      │
└──────────────────────────┬──────────────────────────────────┘
                           │ PIL frames + telemetry dict
┌──────────────────────────▼──────────────────────────────────┐
│            VISION FEATURE EXTRACTION  (HF ENGINE)           │
│  vision_engine.py                                           │
│  • openai/clip-vit-base-patch32  (zero-shot, no training)   │
│  • V_track = [p_dry, p_damp, p_wet, p_drying]               │
│  • CLS-token attention heatmap (Grad-CAM style overlay)     │
│  • Heuristic mock mode — never crashes on cold boot         │
└──────────────────────────┬──────────────────────────────────┘
                           │ probability vector
┌──────────────────────────▼──────────────────────────────────┐
│              MCDA OPTIMIZATION ENGINE                       │
│  mcda_engine.py                                             │
│  • Weighted Sum Model over 4 pit-wall actions               │
│  • A* = argmax_j Σᵢ wᵢ · x_ij                              │
│  • Race-engineer rationale generator                        │
│  • Confidence from score margin between top-2 actions       │
└──────────────────────────┬──────────────────────────────────┘
                           │ strategy call + confidence + rationale
┌──────────────────────────▼──────────────────────────────────┐
│              FRONTEND  (STREAMLIT DASHBOARD)                │
│  app.py                                                     │
│  • Live vision feed + optional Grad-CAM heatmap toggle      │
│  • Classification probability bars + condition badge        │
│  • Live metrics row: condition · confidence · action · frames│
│  • High-visibility MCDA strategy call card                  │
│  • Pit-wall audio alert (autoplay via HTML component)       │
│  • Real-time Plotly temporal trend chart                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Hugging Face Hub Assets

| Asset | Link |
|---|---|
| Vision model (primary) | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) |
| Vision model (alternative) | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) |
| Sample dataset | Publish with `dataset_setup.py`, e.g. `https://huggingface.co/datasets/<your-team>/weather-whiplash-track-frames` |
| Team profiles | `https://huggingface.co/<username>` — one per team member (required by the hackathon rules) |

---

## The MCDA Mathematical Model

The strategy engine scores four candidate actions with a **Weighted Sum Model (WSM)**:

### Actions (A)

| Key | Label |
|---|---|
| A₁ `STAY_OUT` | Maintain current tyres |
| A₂ `PIT_SLICKS` | Box for dry slicks |
| A₃ `PIT_INTERMEDIATES` | Box for intermediate wets |
| A₄ `HOLD_EXTEND_STINT` | Delay the stop 2 laps (traffic risk) |

### Criteria and weights (W)

| # | Criterion | Weight | Signal derivation |
|---|---|---|---|
| w₁ | Dry track probability | **0.35** | p(Dry) + p(Drying) from CLIP |
| w₂ | Lap time falloff | **0.25** | lap delta / 5.0 s (normalized) |
| w₃ | Traffic re-entry penalty | **0.20** | street circuit × traffic density |
| w₄ | Track evaporation rate | **0.10** | (T_track − 10 °C) / 40 °C |
| w₅ | Tyre wear level | **0.10** | wear fraction [0, 1] |
| | **Total** | **1.00** | |

### Scoring formula

Each action **j** receives a utility **x_ij ∈ [0, 1]** on each criterion **i** (e.g. `PIT_SLICKS` gains utility from high dryness, high evaporation, and high wear, and loses it under high traffic risk). The score is:

```
S_j = Σᵢ wᵢ · x_ij       A* = argmax_j S_j
```

Confidence is derived from the score margin between the top two ranked actions:

```
confidence = clamp(0.50 + margin × 2.5,  min=0.50,  max=0.99)
```

Every call ships with a generated race-engineer rationale ("Box this lap, box, confirm") constructed from the live criterion signals.

---

## Running Locally

```bash
git clone <your-repo>
cd weather_whiplash

# Install dependencies (Python 3.10+ recommended)
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

> **Note:** The first launch downloads the CLIP weights (~600 MB from the Hugging Face Hub).
> No GPU required — the engine auto-falls back to CPU. If the model cannot load at all
> (e.g. extreme memory constraints), the app switches to a lightweight heuristic
> pixel-statistics mode and continues to run all features normally.

### Optional: install `accelerate` for faster model loading

```bash
pip install accelerate>=0.26
```

---

## Deploying on Hugging Face Spaces

1. Create a new Space → **Streamlit** SDK (free CPU tier is sufficient).
2. Push all project files to the Space repo — the YAML header in this README
   configures the Space automatically (`sdk_version`, `app_file`, etc.).
3. *(Optional)* Publish the sample dataset:
   ```bash
   huggingface-cli login
   python dataset_setup.py --repo-id <your-team>/weather-whiplash-track-frames
   ```
4. *(Optional)* Drop real track photos into a `sample_frames/` folder in the
   Space repo — the live-stream simulator will use them automatically instead
   of synthetic frames.

---

## Project Files

| File | Purpose |
|---|---|
| [`app.py`](app.py) | Streamlit dashboard — three input modes, audio pit alerts, live metrics row |
| [`vision_engine.py`](vision_engine.py) | CLIP zero-shot classifier, CLS-token attention heatmaps, video sampling |
| [`mcda_engine.py`](mcda_engine.py) | Weighted Sum Model strategy pipeline + race-engineer rationale generator |
| [`trend_engine.py`](trend_engine.py) | Temporal trend analytics — drying velocity, crossover window detection, ETA estimation |
| [`simulation.py`](simulation.py) | Live-stream simulator with synthetic wet→dry frame generator |
| [`dataset_setup.py`](dataset_setup.py) | Uploads the sample dataset to the HF Dataset Hub |
| [`PITCH_PRESENTATION.md`](PITCH_PRESENTATION.md) | 2-minute judge pitch script with per-segment timing and rehearsal checklist |
| [`.streamlit/config.toml`](.streamlit/config.toml) | Streamlit dark theme + HF Spaces server configuration |

---

## Team

**Il Predestinato**

| Name | Hugging Face Profile |
|---|---|
| Krishiv Wahi | [huggingface.co/krishivwahi](https://huggingface.co/astroKW) |

---
