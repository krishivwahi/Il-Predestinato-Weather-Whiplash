"""Temporal trend analytics for the Weather Whiplash detector.

Provides functions to compute **drying velocity** (Δ dry-probability per frame),
detect **crossover windows** (the imminent wet→dry tyre transition moment),
generate **rolling condition label sequences** for pill display, and estimate
the **ETA in laps** until the slick crossover threshold is reached.

All functions are pure — they read the ``history`` list from Streamlit session
state but do not modify it, keeping the analytics layer side-effect free.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Condition labels (must match vision_engine.py)
# ---------------------------------------------------------------------------
CONDITION_LABELS: List[str] = ["Dry", "Damp", "Wet", "Drying"]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
VELOCITY_THRESHOLD_FAST: float = 0.060   # > 6 % / frame  → rapid drying
VELOCITY_THRESHOLD_SLOW: float = 0.025   # > 2.5 % / frame → slow drying
VELOCITY_WET_FAST:       float = -0.060  # < −6 % / frame  → rapid wetting
VELOCITY_WET_SLOW:       float = -0.025  # < −2.5 % / frame → slow wetting

CROSSOVER_VELOCITY_MIN:  float = 0.030   # minimum positive velocity for crossover
CROSSOVER_DRY_LO:        float = 0.30    # lower bound of transition zone
CROSSOVER_DRY_HI:        float = 0.80    # upper bound of transition zone
SLICK_TARGET_DRY:        float = 0.65    # dry-side prob that triggers slick window


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _dry_prob(row: dict) -> float:
    """Return aggregate dry-side probability (Dry + Drying) for a history row."""
    return float(row.get("Dry", 0.0)) + float(row.get("Drying", 0.0))


# ---------------------------------------------------------------------------
# Drying velocity
# ---------------------------------------------------------------------------
def compute_drying_velocity(history: list, window: int = 5) -> float:
    """Compute the average rate of change of dry-side probability per frame.

    Uses a first-difference average over the most-recent *window* frames.
    Positive = track getting drier; negative = track getting wetter.

    Args:
        history: List of classification result dicts from ``st.session_state.history``.
        window:  Number of most-recent frames to use (default 5).

    Returns:
        Float Δ(p_dry + p_drying) / frame, typically in [−0.3, +0.3].
        Returns 0.0 when fewer than 2 frames are available.
    """
    if len(history) < 2:
        return 0.0
    recent = history[-min(window, len(history)):]
    dry_vals = [_dry_prob(r) for r in recent]
    if len(dry_vals) < 2:
        return 0.0
    deltas = [dry_vals[i + 1] - dry_vals[i] for i in range(len(dry_vals) - 1)]
    return sum(deltas) / len(deltas)


# ---------------------------------------------------------------------------
# Trend label
# ---------------------------------------------------------------------------
def trend_label(velocity: float) -> Tuple[str, str, str]:
    """Classify a drying velocity into a human label, indicator symbol, and hex colour.

    Args:
        velocity: Drying velocity from :func:`compute_drying_velocity`.

    Returns:
        ``(label, indicator, colour_hex)`` tuple.
    """
    if velocity > VELOCITY_THRESHOLD_FAST:
        return "RAPID DRYING", "+", "#2ecc71"
    if velocity > VELOCITY_THRESHOLD_SLOW:
        return "DRYING",       "+", "#7ed6df"
    if velocity < VELOCITY_WET_FAST:
        return "RAPID WETTING","-", "#e74c3c"
    if velocity < VELOCITY_WET_SLOW:
        return "WETTING",      "-", "#f39c12"
    return     "STABLE",       "=", "#8b949e"


# ---------------------------------------------------------------------------
# Crossover window detection
# ---------------------------------------------------------------------------
def detect_crossover_window(history: list, window: int = 5) -> bool:
    """Return True when a wet→dry tyre crossover window is active.

    A crossover window is open when **both**:

    1. Drying velocity ≥ :data:`CROSSOVER_VELOCITY_MIN` (track actively drying), **and**
    2. Current dry-side probability is inside the transition zone
       [``CROSSOVER_DRY_LO``, ``CROSSOVER_DRY_HI``]
       (not yet fully dry, but clearly headed there).

    Args:
        history: List of classification result dicts.
        window:  Frame window for velocity calculation.

    Returns:
        ``True`` if the slick crossover window is imminently open.
    """
    if len(history) < 3:
        return False
    velocity    = compute_drying_velocity(history, window=window)
    current_dry = _dry_prob(history[-1])
    return (
        velocity > CROSSOVER_VELOCITY_MIN
        and CROSSOVER_DRY_LO < current_dry < CROSSOVER_DRY_HI
    )


# ---------------------------------------------------------------------------
# ETA estimation
# ---------------------------------------------------------------------------
def compute_eta_laps(
    history: list,
    target_dry: float = SLICK_TARGET_DRY,
    window: int = 5,
) -> Optional[float]:
    """Estimate the number of laps until dry-side probability reaches *target_dry*.

    Uses a linear extrapolation of the current drying velocity.

    Args:
        history:    List of classification result dicts.
        target_dry: Dry-side probability threshold (default :data:`SLICK_TARGET_DRY`).
        window:     Frame window for velocity calculation.

    Returns:
        Estimated laps (frames) as a float, or ``None`` when:
        - Fewer than 2 frames available.
        - Velocity ≤ 0 (track not drying).
        - Target already exceeded.
    """
    if len(history) < 2:
        return None
    current  = _dry_prob(history[-1])
    if current >= target_dry:
        return None
    velocity = compute_drying_velocity(history, window=window)
    if velocity <= 1e-6:
        return None
    return round((target_dry - current) / velocity, 1)


# ---------------------------------------------------------------------------
# Rolling condition sequence
# ---------------------------------------------------------------------------
def get_condition_sequence(history: list, n: int = 10) -> List[Tuple[str, str]]:
    """Return the last *n* ``(frame_label, top_condition)`` pairs for pill display.

    Args:
        history: List of classification result dicts.
        n:       Maximum number of entries to return (oldest first).

    Returns:
        List of ``(frame_label, top_condition)`` tuples.
    """
    recent = history[-n:] if len(history) > n else history
    result: List[Tuple[str, str]] = []
    for row in recent:
        probs = {label: float(row.get(label, 0.0)) for label in CONDITION_LABELS}
        top   = max(probs, key=probs.get)  # type: ignore[arg-type]
        result.append((str(row.get("frame", "?")), top))
    return result
