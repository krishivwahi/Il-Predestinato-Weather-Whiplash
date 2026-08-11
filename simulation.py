"""Simulated live-stream frame pipeline for the Weather Whiplash detector.

Provides :class:`TrackStreamSimulator`, which yields a sequence of track
frames representing a wet → drying → dry transition (Lap 1: heavy wet,
Lap 10: drying line, Lap 15: dry line) together with simulated telemetry
consistent with the surface state.

If a local ``sample_frames/`` directory containing real track images exists,
the simulator maps those images onto the wetness schedule in order.
Otherwise a synthetic PIL image generator produces colour-coded asphalt
frames with wet-glare puddles, a drying racing line, and kerb stripes so
the app **always** has data to run on — even with no local images and no
internet connection.
"""

from __future__ import annotations

import os
from typing import Dict, Generator, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Compatible resampling constant
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* into the closed interval [low, high]."""
    return max(low, min(high, value))


class TrackStreamSimulator:
    """Generator of ``(frame_id, image, telemetry)`` tuples for a race sequence.

    The wetness schedule eases from 1.0 (heavy wet) on frame 0 down to 0.0
    (fully dry racing line) on the last frame, with mild Gaussian noise added
    so the trend chart looks organic rather than perfectly linear.

    Args:
        total_frames: Number of simulated laps/frames to generate (≥ 2).
        frame_size: ``(width, height)`` in pixels for synthetic frames.
        sample_dir: Directory containing real track images to use instead of
            synthetic frames if present.
        seed: Random seed for reproducible frame sequences.
    """

    def __init__(
        self,
        total_frames: int = 15,
        frame_size: Tuple[int, int] = (640, 360),
        sample_dir: str = "sample_frames",
        seed: int = 7,
    ) -> None:
        self.total_frames = max(2, int(total_frames))
        self.frame_size = frame_size
        self.sample_dir = sample_dir
        self.rng = np.random.default_rng(seed)
        self._sample_paths: List[str] = self._discover_samples()

    # ------------------------------------------------------------------
    # Sample image discovery
    # ------------------------------------------------------------------
    def _discover_samples(self) -> List[str]:
        """Return sorted image paths from ``sample_dir`` if any exist."""
        if not os.path.isdir(self.sample_dir):
            return []
        paths = [
            os.path.join(self.sample_dir, name)
            for name in sorted(os.listdir(self.sample_dir))
            if name.lower().endswith(VALID_EXTENSIONS)
        ]
        return paths

    # ------------------------------------------------------------------
    # Wetness schedule + telemetry
    # ------------------------------------------------------------------
    def _wetness_at(self, index: int) -> float:
        """Compute wetness in [0, 1] for frame *index* (1.0 = heavy wet)."""
        progress = index / max(1, self.total_frames - 1)
        base = 1.0 - progress ** 1.15        # slight acceleration toward dry
        noise = float(self.rng.normal(0.0, 0.03))
        return _clamp(base + noise)

    def _telemetry_at(self, index: int, wetness: float) -> Dict[str, float]:
        """Return simulated telemetry consistent with current surface wetness.

        Returns:
            Dict with keys ``lap``, ``lap_delta``, ``track_temp_c``,
            ``tire_wear_pct``, ``grip_estimate``, and ``wetness_truth``.
        """
        progress = index / max(1, self.total_frames - 1)
        lap_delta = round(
            _clamp(
                0.3 + wetness * 4.2 + float(self.rng.normal(0.0, 0.15)),
                0.0,
                5.0,
            ),
            2,
        )
        return {
            "lap":           index + 1,
            "lap_delta":     lap_delta,
            "track_temp_c":  round(16.0 + (1.0 - wetness) * 18.0, 1),
            "tire_wear_pct": round(min(0.95, 0.12 + progress * 0.70), 2),
            "grip_estimate": round(1.0 - wetness * 0.6, 2),
            "wetness_truth": round(wetness, 2),
            # Humidity tracks wetness: 90% on a soaking lap 1, down to ~38% when dry.
            "humidity_pct":  round(max(38.0, 38.0 + wetness * 52.0), 1),
        }

    # ------------------------------------------------------------------
    # Synthetic frame generation
    # ------------------------------------------------------------------
    def _synthetic_frame(self, wetness: float) -> Image.Image:
        """Render a colour-coded asphalt frame with wet-glare patterns.

        Visual cues (all driven by the *wetness* scalar in [0, 1]):

        * **Asphalt darkness** — wet track is darker than dry asphalt.
        * **Blue-grey wet film** — semi-transparent overlay scaled by wetness.
        * **Specular puddles** — elliptical glare patches with realistic alpha.
        * **Drying racing line** — lighter central groove with tyre-track marks.
        * **Kerb stripes** — red/white kerb at the top edge for visual context.

        Args:
            wetness: Surface wetness in [0, 1]. 1.0 = standing water.

        Returns:
            A PIL RGB image of size ``self.frame_size``.
        """
        w, h = self.frame_size

        # --- Asphalt base with granular noise --------------------------------
        base_gray = int(75 + (1.0 - wetness) * 35)   # wet=darker, dry=lighter
        base = np.full((h, w, 3), base_gray, dtype=np.int16)
        noise = self.rng.integers(-16, 16, (h, w, 1), dtype=np.int16)
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, "RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        # --- Red/white kerb stripes along the top edge -----------------------
        stripe_w = 40
        for x in range(0, w, stripe_w * 2):
            draw.rectangle([x, 0, x + stripe_w, 20], fill=(210, 30, 30, 255))
            draw.rectangle([x + stripe_w, 0, x + stripe_w * 2, 20], fill=(235, 235, 235, 255))

        # --- Wet film overlay -------------------------------------------------
        if wetness > 0.05:
            draw.rectangle(
                [0, 21, w, h],
                fill=(18, 26, 58, int(115 * wetness)),
            )
            # Specular glare puddles — count and size grow with wetness
            n_puddles = int(2 + wetness * 12)
            for _ in range(n_puddles):
                cx = int(self.rng.integers(20, w - 20))
                cy = int(self.rng.integers(h // 3, h - 20))
                rx = int(16 + self.rng.integers(0, 60) * wetness)
                ry = max(5, rx // 4)
                alpha = int(65 + 130 * wetness)
                draw.ellipse(
                    [cx - rx, cy - ry, cx + rx, cy + ry],
                    fill=(210, 222, 245, alpha),
                )
                # Inner bright highlight
                draw.ellipse(
                    [cx - rx // 3, cy - ry // 3, cx + rx // 3, cy + ry // 3],
                    fill=(240, 248, 255, min(255, alpha + 40)),
                )

        # --- Drying racing line ----------------------------------------------
        if wetness < 0.88:
            dry_alpha = int(_clamp(1.0 - wetness / 0.88) * 170)
            line_top    = int(h * 0.44)
            line_bottom = int(h * 0.73)
            # The main dry groove
            draw.rectangle(
                [0, line_top, w, line_bottom],
                fill=(118, 114, 108, dry_alpha),
            )
            # Two tyre-groove marks inside the line
            for gy in (line_top + 16, line_bottom - 16):
                draw.line(
                    [(0, gy), (w, gy)],
                    fill=(145, 140, 132, dry_alpha),
                    width=7,
                )

        # Subtle Gaussian blur for realism
        return img.filter(ImageFilter.GaussianBlur(radius=1.2))

    # ------------------------------------------------------------------
    # Frame sourcing
    # ------------------------------------------------------------------
    def _frame_at(self, index: int, wetness: float) -> Image.Image:
        """Return a real sample frame if available, else a synthetic one.

        When real images are present the schedule is mapped linearly onto the
        available files: the wettest images come first, the driest come last.
        """
        if self._sample_paths:
            pos = int(
                round(index / max(1, self.total_frames - 1) * (len(self._sample_paths) - 1))
            )
            try:
                img = Image.open(self._sample_paths[pos]).convert("RGB")
                return img.resize(self.frame_size, _LANCZOS)
            except OSError:
                pass  # unreadable file → fall through to synthetic
        return self._synthetic_frame(wetness)

    # ------------------------------------------------------------------
    # Public stream interface
    # ------------------------------------------------------------------
    def stream(self) -> Generator[Tuple[int, Image.Image, Dict[str, float]], None, None]:
        """Yield ``(frame_id, PIL image, telemetry dict)`` for each simulated lap.

        ``frame_id`` is 1-based (lap number). Telemetry keys mirror the
        sidebar telemetry controls in ``app.py`` (``lap_delta``,
        ``track_temp_c``, ``tire_wear_pct``, …).
        """
        for index in range(self.total_frames):
            wetness  = self._wetness_at(index)
            image    = self._frame_at(index, wetness)
            telemetry = self._telemetry_at(index, wetness)
            yield index + 1, image, telemetry

    def frame_for_wetness(self, wetness: float) -> Image.Image:
        """Public helper: render a single synthetic frame at an arbitrary wetness.

        Useful for ``dataset_setup.py`` to generate per-category samples.

        Args:
            wetness: Wetness in [0, 1]. Clamped internally.

        Returns:
            A PIL RGB image of size ``self.frame_size``.
        """
        return self._synthetic_frame(_clamp(wetness))
