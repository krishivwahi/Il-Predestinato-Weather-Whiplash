"""Multi-Criteria Decision Analysis (MCDA) strategy pipeline.

Implements a Weighted Sum Model (WSM) over four candidate pit-wall actions,
scoring each against five weighted criteria derived from computer vision
probabilities, telemetry, track-context inputs, and **weather humidity**.

Mathematical model:

    S_j = Σᵢ( wᵢ · x_ij )      A* = argmax_j S_j

Actions (A):
    A₁  STAY_OUT           — maintain current tyres
    A₂  PIT_SLICKS         — box for dry slicks
    A₃  PIT_INTERMEDIATES  — box for intermediate wets
    A₄  HOLD_EXTEND_STINT  — delay the stop 2 laps (traffic risk)

Criteria (W, must sum to 1.0):
    w₁ = 0.35  dry_probability   : p(Dry) + p(Drying) from CLIP
    w₂ = 0.25  lap_time_falloff  : normalized pace loss over 5 s baseline
    w₃ = 0.20  traffic_penalty   : re-entry risk (street circuit × traffic density)
    w₄ = 0.10  evaporation_rate  : temp_factor × (0.30 + 0.70 × (1 − humidity))
                                   High humidity suppresses drying even at high T.
    w₅ = 0.10  tire_wear         : current tyre wear fraction [0, 1]
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
ACTIONS: Dict[str, str] = {
    "STAY_OUT":          "Stay Out — maintain current tyres",
    "PIT_SLICKS":        "Box Box — fit Dry Slicks",
    "PIT_INTERMEDIATES": "Box Box — fit Intermediates",
    "HOLD_EXTEND_STINT": "Hold — extend stint 2 laps (traffic risk)",
}

# ---------------------------------------------------------------------------
# Criteria weights (sum = 1.0)
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    "dry_probability":  0.35,  # w₁ — vision signal: p(Dry) + p(Drying)
    "lap_time_falloff": 0.25,  # w₂ — normalized pace loss [0, 1]
    "traffic_penalty":  0.20,  # w₃ — pit re-entry risk [0, 1]
    "evaporation_rate": 0.10,  # w₄ — track temperature drying potential
    "tire_wear":        0.10,  # w₅ — current tyre wear fraction [0, 1]
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "WEIGHTS must sum to 1.0"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* into the closed interval [low, high]."""
    return max(low, min(high, value))


def _build_rationale(
    action: str,
    dryness: float,
    wetness: float,
    lap_delta: float,
    track_temp_c: float,
    tire_wear_pct: float,
    is_street_circuit: bool,
    traffic_dense: bool,
    humidity_pct: float = 50.0,
) -> str:
    """Compose a race-engineer style textual rationale for the chosen action.

    Args:
        action: One of the four ``ACTIONS`` keys.
        dryness: Aggregated dry-side probability (Dry + Drying).
        wetness: Aggregated wet-side probability (Wet + 0.5×Damp).
        lap_delta: Pace loss in seconds vs. dry-weather baseline.
        track_temp_c: Track surface temperature in °C.
        tire_wear_pct: Current tyre wear in [0, 1].
        is_street_circuit: True for street circuits.
        traffic_dense: True when caught in a midfield train.
        humidity_pct: Ambient relative humidity in percent (0–100).

    Returns:
        A single-paragraph rationale string for display on the strategy card.
    """
    circuit = "street circuit" if is_street_circuit else "permanent circuit"
    traffic = "dense midfield traffic" if traffic_dense else "clear air"

    if action == "PIT_SLICKS":
        return (
            f"Surface dryness confidence is {dryness:.0%}. Track temp {track_temp_c:.0f} °C "
            f"with {humidity_pct:.0f}% humidity — evaporation rate is "
            f"{'high' if humidity_pct < 40 else 'moderate' if humidity_pct < 70 else 'suppressed'}. "
            f"Pace loss is {lap_delta:+.1f} s per lap — the slick crossover window is open. "
            f"On a {circuit} with {traffic}, we take the undercut now before rivals react. "
            f"Box this lap, box, confirm."
        )
    if action == "PIT_INTERMEDIATES":
        return (
            f"Vision engine reads {wetness:.0%} wet-side probability. Humidity at "
            f"{humidity_pct:.0f}% confirms slow evaporation — the surface is not drying soon. "
            f"Pace loss of {lap_delta:+.1f} s says current rubber has no grip. "
            f"Standing water risk outweighs pit-loss cost. Box for intermediates — "
            f"safety and lap time point the same way. Box, box, intermediates."
        )
    if action == "HOLD_EXTEND_STINT":
        return (
            f"Conditions are at the crossover ({dryness:.0%} dry-side confidence). "
            f"Humidity is {humidity_pct:.0f}% — track evaporation is "
            f"{'progressing well' if humidity_pct < 55 else 'slow, buy more time'}. "
            f"On a {circuit} with {traffic}, pitting now sacrifices track position. "
            f"Target +2 laps: hold, manage the tyres ({tire_wear_pct:.0%} worn), "
            f"and re-evaluate every lap. Stay out, stay out."
        )
    # STAY_OUT
    return (
        f"Pace delta of {lap_delta:+.1f} s is within tolerance, tyre wear at "
        f"{tire_wear_pct:.0%}, and humidity of {humidity_pct:.0f}% means "
        f"{'conditions are stable' if humidity_pct > 60 else 'evaporation is progressing'}. "
        f"No stop justified — track position is worth more than a marginal compound "
        f"change right now. Stay out and feed us surface reports every lap."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def calculate_mcda_action(
    cv_probs: Dict[str, float],
    lap_delta: float,
    is_street_circuit: bool,
    traffic_dense: bool,
    track_temp_c: float,
    tire_wear_pct: float,
    humidity_pct: float = 50.0,
) -> Dict[str, Any]:
    """Run the Weighted Sum Model and return the optimal strategy call.

    Args:
        cv_probs: Vision probabilities with keys ``"Dry"``, ``"Damp"``,
            ``"Wet"``, ``"Drying"`` (values sum to ~1.0).
        lap_delta: Pace loss vs. dry baseline in seconds (e.g. ``+2.5``).
        is_street_circuit: ``True`` for street circuits (high re-entry penalty).
        traffic_dense: ``True`` when the car is caught in a midfield train.
        track_temp_c: Track surface temperature in °C (10–50 typical).
        tire_wear_pct: Current tyre wear fraction in ``[0.0, 1.0]``.
        humidity_pct: Ambient relative humidity in percent (0–100).
            High humidity suppresses evaporation, reducing the w₄ criterion score.

    Returns:
        A dict containing:

        - ``recommended_action`` (str): Key from :data:`ACTIONS`.
        - ``recommended_label`` (str): Human-readable action label.
        - ``confidence_score`` (float): Score margin–based confidence in [0.50, 0.99].
        - ``scores_breakdown`` (dict): Raw utility matrix and criterion signals.
        - ``rationale`` (str): Race-engineer style textual justification.
    """
    # ── Criterion signals (all normalized to [0, 1]) ────────────────────────
    dryness    = _clamp(cv_probs.get("Dry", 0.0) + cv_probs.get("Drying", 0.0))
    wetness    = _clamp(cv_probs.get("Wet", 0.0) + 0.5 * cv_probs.get("Damp", 0.0))
    pace_loss  = _clamp(lap_delta / 5.0)

    if is_street_circuit and traffic_dense:
        traffic_risk = 1.0
    elif traffic_dense:
        traffic_risk = 0.6
    elif is_street_circuit:
        traffic_risk = 0.35
    else:
        traffic_risk = 0.10

    # w₄ evaporation: temperature drives drying; humidity suppresses it.
    # Formula: temp_factor × (0.30 + 0.70 × (1 − humidity_factor))
    # → 50 °C / 0 % humidity  → 1.00 (maximum drying)
    # → 50 °C / 100 % humidity → 0.30 (high temp but saturated air)
    # → 10 °C / any humidity   → 0.00 (no thermal drying potential)
    temp_factor     = _clamp((track_temp_c - 10.0) / 40.0)
    humidity_factor = _clamp(humidity_pct / 100.0)
    evaporation     = _clamp(temp_factor * (0.30 + 0.70 * (1.0 - humidity_factor)))
    wear            = _clamp(tire_wear_pct)

    # Peaks at 1.0 at the wet→dry crossover (dryness ≈ 0.5);
    # rewards HOLD for staying tactically flexible.
    transition = 1.0 - abs(dryness - 0.5) * 2.0

    # ── Utility matrix x_ij ─────────────────────────────────────────────────
    # Each entry is a per-criterion utility in [0, 1] for the corresponding action.
    utilities: Dict[str, Dict[str, float]] = {
        "STAY_OUT": {
            "dry_probability":  1.0 - transition * 0.6 - wetness * 0.3,
            "lap_time_falloff": 1.0 - pace_loss,
            "traffic_penalty":  traffic_risk,          # staying out avoids re-entry
            "evaporation_rate": 0.5,
            "tire_wear":        1.0 - wear,
        },
        "PIT_SLICKS": {
            "dry_probability":  dryness,
            "lap_time_falloff": pace_loss,
            "traffic_penalty":  1.0 - traffic_risk,    # pitting incurs re-entry
            "evaporation_rate": evaporation,
            "tire_wear":        wear,
        },
        "PIT_INTERMEDIATES": {
            "dry_probability":  wetness,
            "lap_time_falloff": pace_loss,
            "traffic_penalty":  1.0 - traffic_risk,
            "evaporation_rate": 1.0 - evaporation,
            "tire_wear":        wear,
        },
        "HOLD_EXTEND_STINT": {
            "dry_probability":  transition,
            "lap_time_falloff": 1.0 - pace_loss * 0.5,
            "traffic_penalty":  traffic_risk,
            "evaporation_rate": evaporation * 0.7,
            "tire_wear":        1.0 - wear * 0.8,
        },
    }

    # ── Weighted Sum Model  S_j = Σᵢ wᵢ · x_ij ─────────────────────────────
    scores: Dict[str, float] = {
        action: round(
            sum(WEIGHTS[c] * _clamp(x) for c, x in criteria.items()), 4
        )
        for action, criteria in utilities.items()
    }

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_action, best_score = ranked[0]
    score_margin = best_score - ranked[1][1]

    # Confidence is a function of the score margin between top two actions.
    confidence = _clamp(0.50 + score_margin * 2.5, 0.50, 0.99)

    rationale = _build_rationale(
        best_action,
        dryness,
        wetness,
        lap_delta,
        track_temp_c,
        tire_wear_pct,
        is_street_circuit,
        traffic_dense,
        humidity_pct,
    )

    return {
        "recommended_action": best_action,
        "recommended_label":  ACTIONS[best_action],
        "confidence_score":   round(confidence, 3),
        "scores_breakdown": {
            "action_scores": scores,
            "criteria_signals": {
                "dry_probability":  round(dryness,      3),
                "lap_time_falloff": round(pace_loss,    3),
                "traffic_penalty":  round(traffic_risk, 3),
                "evaporation_rate": round(evaporation,    3),
                "humidity_pct":     round(humidity_pct,   1),
                "tire_wear":        round(wear,           3),
            },
            "weights": WEIGHTS,
        },
        "rationale": rationale,
    }
