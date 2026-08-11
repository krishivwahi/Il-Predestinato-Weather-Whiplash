# Weather Whiplash — Executive Pitch Script

**Target audience:** Executive Board & Technical Evaluators
**Total duration:** ~2 minutes (120 seconds)
**Presenters:** 2 — **Race Strategist** (S) and **Lead AI Engineer** (E)
**Pre-demo checklist:** Application initialized, heatmap overlay toggle OFF, simulation set to 15 laps, circuit = Street Circuit, traffic = ON

---

## Segment 1 — Technical & Strategic Problem (0:00 – 0:25)

**[Visual: Onboard video stream of wet circuit surface; dynamic weather transition]**

> **(S):** "Track surface conditions on high-speed circuits evolve far faster than traditional meteorology models can report. Operating on sub-optimal tyre compounds incurs a 2 to 4 second penalty per lap. On street circuits with constrained overtaking zones, premature pit windows drop vehicles into dense midfield traffic. Weather Whiplash automates track surface assessment and compound crossover decision-making in real time."

---

## Segment 2 — Computer Vision & Machine Learning Architecture (0:25 – 0:55)

**[Visual: Ingestion of track surface frame; visual probability classification vector rendering]**

> **(E):** "Weather Whiplash ingests optical video feeds into a **zero-shot CLIP vision model (`openai/clip-vit-base-patch32`) hosted on Hugging Face**. The pipeline evaluates surface characteristics across four discrete classifications: Dry, Damp, Wet, and Drying.

**[Action: Enable Grad-CAM Heatmap overlay]**

> **(E):** "To ensure model explainability for race engineers, the CLS-token attention heatmap visualizes key visual signals — distinguishing standing water film from clearing racing grooves."

---

## Segment 3 — Decision Support & MCDA Strategy Engine (0:55 – 1:25)

**[Visual: Sidebar configured for Street Circuit + Dense Traffic; real-time MCDA decision output updates]**

> **(S):** "Computer vision is coupled with multi-criteria operational constraints. Our MCDA engine evaluates a Weighted Sum Model across four strategy actions, balancing surface dryness probability, lap time delta, traffic re-entry penalty, evaporation rate, and tire wear.

**[Action: Demonstrate real-time parameter adjustment]**

> **(S):** "When traffic density presents high re-entry risk, the system recommends stint extension. As surface drying accelerates and pace loss exceeds threshold, the decision engine signals pit entry."

---

## Segment 4 — Real-time Simulation & Analytical Output (1:25 – 2:00)

**[Action: Click 'Start Live Simulation Stream' — 15-lap simulation sequence]**

> **(E):** "During real-time telemetry streaming, surface state evolution is tracked lap-by-lap. As dry probability crosses the crossover threshold and evaporation rate increases, the strategy engine triggers the compound change recommendation with confidence metrics and engineer rationales."

**[Closing]**

> **(S + E):** "Weather Whiplash provides scalable, real-time decision intelligence for dynamic racing strategy. Thank you."

---

## Delivery Notes

### Per-segment timing cues

| Segment | Start | End | Key visual |
|---------|-------|-----|-----------|
| 1 – The Problem | 0:00 | 0:25 | Wet track footage / no dashboard yet |
| 2 – Vision Engine | 0:25 | 0:55 | Upload wet frame → bars animate → heatmap reveal |
| 3 – Strategy Engine | 0:55 | 1:25 | Sidebar: Street Circuit + Dense Traffic ON |
| 4 – Live Demo | 1:25 | 2:00 | Start Stream button → 15-lap run begins |

### Rehearsal checklist
- [ ] Run a full 15-lap stream the night before to confirm the slick call lands around lap 12–13.
- [ ] Keep the Grad-CAM toggle **OFF** when the page loads — flip it live in Segment 2 for maximum impact.
- [ ] Test the pit-wall audio alert on the actual presentation machine (some browsers block autoplay without a prior user gesture — click anything on the page first).
- [ ] If venue Wi-Fi is unreliable, the heuristic mock mode still drives the complete demo offline — no model download required. Set `MOCK_DEMO=1` and restart `streamlit run app.py` to force mock mode.

### Presenter tips
- Segment 1 is delivered *before* opening the dashboard tab — the problem statement should land with the audience before they see any UI.
- Segment 3: move the Lap Delta slider from 2.5 s to 0.5 s while talking — the action card switching to STAY_OUT in real time is a powerful live demonstration of the MCDA engine reacting.
- End on "Thank you" as the trend chart's Dry line peaks — the visual metaphor writes itself.
