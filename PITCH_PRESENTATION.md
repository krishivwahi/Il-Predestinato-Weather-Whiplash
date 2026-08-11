# 🎬 Weather Whiplash — 2-Minute Demo Pitch Script

**Target audience:** Hackathon judges
**Total duration:** ~2 minutes (120 seconds)
**Presenters:** 2 — **Strategist** (S) and **Engineer** (E)
**Pre-demo checklist:** App running, heatmap toggle OFF, stream set to 15 laps, circuit = Street Circuit, traffic = ON

---

## Segment 1 — The Problem (0:00 – 0:25)

**[Visual: onboard footage of a wet street circuit; rain on visor; wipers on the pit wall camera]**

> **(S):** "Rain hits a street circuit and the track can change from Wet to Dry in six laps —
> faster than any weather radar update. Every lap on the wrong tyre costs two to four seconds.
> But on a street track there is nowhere to overtake: pitting one lap too early drops you into
> traffic you can never pass, and the win is gone. Today that crossover call is a human
> squinting at a television feed and trusting their gut. We automated it — in real time."

---

## Segment 2 — The AI Vision Engine (0:25 – 0:55)

**[Visual: upload a wet track frame; watch the four probability bars animate]**

> **(E):** "Weather Whiplash feeds trackside frames straight into
> **CLIP — `openai/clip-vit-base-patch32` — pulled live from the Hugging Face Hub**.
> Zero-shot, zero fine-tuning: the model scores every frame against four natural-language
> descriptions — Dry, Damp, Wet, Drying — and returns a probability vector in under a second."

**[Action: flip the Grad-CAM toggle ON]**

> **(E):** "And we do not ask you to trust a black box. The CLS-token attention heatmap
> lights up exactly what the model is focusing on — the shiny standing water off the racing
> line, the lighter dry groove forming where the rubber has cleared. That is explainability
> a race engineer can verify at a glance."

---

## Segment 3 — The Strategy Engine (0:55 – 1:25)

**[Visual: sidebar set to Street Circuit + Dense Traffic; watch the strategy card change]**

> **(S):** "Vision alone does not win races — context does. Our MCDA engine runs a
> Weighted Sum Model across four pit-wall actions, blending five weighted criteria:
> **dry-track probability at 35%**, lap-time falloff at 25%, traffic re-entry penalty at 20%,
> track evaporation and tyre wear at 10% each.
>
> On a street circuit with dense traffic, pitting costs track position you cannot recover —
> so the engine calls **HOLD**, and tells you why in plain language the whole pit wall understands.
> The moment the crossover math flips — dry probability climbs, pace loss exceeds tolerance —
> the card goes orange, the pit alert fires, and you hear it: **Box, box. Fit slicks.**"

---

## Segment 4 — Live Demo Walkthrough (1:25 – 2:00)

**[Action: hit '🚦 Start / Restart Stream' — 15-lap wet-to-dry transition runs at 1.5 s/frame]**

> **(E):** "Watch a full race evolve live. Lap one: heavy wet — vision reads 84% Wet,
> engine calls intermediates. By lap eight the Drying line climbs on the trend chart,
> the traffic penalty holds us in —HOLD, extend 2 laps. Lap twelve: dry probability
> crosses the threshold, 34 degrees of evaporation — the card flips orange,
> the chirp fires — **Box for Slicks, high confidence**, with the rationale a real
> race engineer would radio. One frame every 1.5 seconds, live charts, live calls,
> all running free on a Hugging Face Space CPU."

**[Closing — both presenters face camera]**

> **(S + E):** "Streamlit frontend, CLIP from the Hub, our sample dataset on the
> Dataset Hub, every teammate on their own Hugging Face account.
> **Weather Whiplash** — because the team that reads the track first wins. Thank you."

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
