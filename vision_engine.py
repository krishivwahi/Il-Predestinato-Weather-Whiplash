"""Hugging Face vision inference engine for track condition detection.

Provides :class:`TrackVisionEngine`, which wraps the ``openai/clip-vit-base-patch32``
zero-shot CLIP model to classify track surface frames into four states
(Dry, Damp, Wet, Drying), generate CLS-token attention heatmaps for
explainability, and sample frames from uploaded video files.

Graceful degradation: if the model cannot be loaded (e.g. memory-constrained
Hugging Face Space cold boot), the engine flips into ``mock_mode`` and uses
lightweight pixel-statistics heuristics so the app **never** crashes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = "openai/clip-vit-base-patch32"

CONDITION_LABELS: List[str] = ["Dry", "Damp", "Wet", "Drying"]

# Zero-shot candidate text prompts aligned with each label.
_PROMPTS: Dict[str, str] = {
    "Dry":    "a dry asphalt racing track",
    "Wet":    "a wet racetrack with standing water",
    "Damp":   "a damp surface track",
    "Drying": "a drying racetrack line with dry tyre grooves",
}

# Pillow resampling compatibility (Pillow ≥ 10 uses Image.Resampling enum)
try:
    _BICUBIC = Image.Resampling.BICUBIC
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    _BICUBIC = Image.BICUBIC  # type: ignore[attr-defined]
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


def _get_cmap(name: str = "jet"):
    """Return a matplotlib colormap, compatible with all matplotlib versions."""
    try:
        from matplotlib import colormaps
        return colormaps[name]
    except (ImportError, KeyError):
        from matplotlib import cm  # type: ignore[import]
        return cm.get_cmap(name)


class TrackVisionEngine:
    """Zero-shot CLIP classifier + explainability heatmaps for track frames.

    Attributes:
        model_id: Hugging Face model identifier used for loading.
        mock_mode: If ``True`` the engine uses heuristic-only inference.
        device: PyTorch device string (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id: str = model_id
        self.mock_mode: bool = False
        self.device: str = "cpu"
        self.model = None
        self.processor = None
        self._load_model()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Attempt to load the CLIP model; flip to mock_mode on any failure."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading CLIP model %s on %s …", self.model_id, self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
            self.model = CLIPModel.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()
            logger.info("CLIP model ready on %s.", self.device)
        except Exception as exc:  # noqa: BLE001 — any failure → mock mode
            logger.warning(
                "CLIP model load failed (%s). Activating heuristic mock mode.", exc
            )
            self.mock_mode = True

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify_track(self, image: Image.Image) -> Dict[str, float]:
        """Classify a track frame into normalized condition probabilities.

        Args:
            image: A PIL RGB image of the track surface.

        Returns:
            ``{"Dry": float, "Damp": float, "Wet": float, "Drying": float}``
            with values summing to approximately 1.0.
        """
        if self.mock_mode or self.model is None or self.processor is None:
            return self._heuristic_probs(image)
        try:
            import torch

            texts = [_PROMPTS[label] for label in CONDITION_LABELS]
            inputs = self.processor(
                text=texts, images=image, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()
            return {label: float(p) for label, p in zip(CONDITION_LABELS, probs)}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inference failed (%s). Using heuristic fallback.", exc)
            return self._heuristic_probs(image)

    def _heuristic_probs(self, image: Image.Image) -> Dict[str, float]:
        """Pixel-statistics fallback classifier (no model required).

        Derives wet/dry signal from brightness, specular glare, and blue-channel
        bias — reliable enough to drive the MCDA engine in demo conditions.
        """
        arr = (
            np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32)
            / 255.0
        )
        brightness = float(arr.mean())
        glare = float((arr.max(axis=2) > 0.85).mean())
        blue_bias = float(arr[..., 2].mean() - arr[..., 0].mean())

        wet_score = max(
            0.05,
            glare * 5.0 + max(blue_bias, 0.0) * 4.0 + max(0.45 - brightness, 0.0) * 2.0,
        )
        dry_score = max(0.05, (brightness - 0.35) * 2.0 - glare * 3.0)
        damp_score = max(0.05, 0.6 - abs(wet_score - dry_score))
        drying_score = max(0.05, min(wet_score, dry_score) * 1.5)

        raw = {
            "Dry":    dry_score,
            "Damp":   damp_score,
            "Wet":    wet_score,
            "Drying": drying_score,
        }
        total = sum(raw.values())
        return {label: value / total for label, value in raw.items()}

    # ------------------------------------------------------------------
    # Explainability — attention heatmap
    # ------------------------------------------------------------------
    def generate_attention_heatmap(self, image: Image.Image) -> Image.Image:
        """Return the input image blended with a Grad-CAM style heatmap.

        In model mode the map is derived from CLIP's vision-transformer CLS-token
        attention averaged over the last four encoder layers. In mock mode a
        glare/darkness heuristic highlights wet patches from pixel statistics.

        Args:
            image: Original PIL RGB track frame.

        Returns:
            A new PIL RGB image with the heatmap blended at 45% opacity.
        """
        heat: Optional[np.ndarray] = None
        if not (self.mock_mode or self.model is None):
            try:
                heat = self._clip_attention_map(image)
            except Exception as exc:  # noqa: BLE001 — heatmap is cosmetic
                logger.warning("Attention map failed (%s). Heuristic fallback.", exc)
        if heat is None:
            heat = self._heuristic_heatmap(image)
        return self._blend_heatmap(image, heat)

    def _clip_attention_map(self, image: Image.Image) -> np.ndarray:
        """Derive a 2D attention grid from CLIP's ViT CLS-token row.

        Returns a float32 array in [0, 1] shaped (grid_h, grid_w).
        """
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            out = self.model.vision_model(
                pixel_values=pixel_values,
                output_attentions=True,
            )

        # Guard: attentions may be None if the model config suppresses them.
        if out.attentions is None or len(out.attentions) == 0:
            logger.warning("CLIP returned no attention tensors; using heuristic.")
            return self._heuristic_heatmap(image)

        # Average the last ≤4 layers and all heads for a stable, smooth map.
        n_layers = len(out.attentions)
        layers_to_use = out.attentions[max(0, n_layers - 4):]
        att = torch.stack(layers_to_use).mean(dim=0).mean(dim=1)[0]  # (seq, seq)

        # CLS (index 0) → patch tokens (index 1:)
        cls_to_patch = att[0, 1:]
        n = int(round(cls_to_patch.numel() ** 0.5))
        if n * n != cls_to_patch.numel():
            # Patch count is not a perfect square (e.g. non-square input) — truncate.
            cls_to_patch = cls_to_patch[: n * n]

        grid = cls_to_patch.reshape(n, n).float().cpu().numpy()
        lo, hi = float(grid.min()), float(grid.max())
        return (grid - lo) / (hi - lo + 1e-8)

    def _heuristic_heatmap(self, image: Image.Image) -> np.ndarray:
        """Highlight specular glare and dark wet film without a model.

        Returns a float32 array in [0, 1] shaped (224, 224).
        """
        arr = (
            np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
            / 255.0
        )
        # High luminance → specular water glare
        glare = np.clip((arr.max(axis=2) - 0.75) * 4.0, 0.0, 1.0)
        # Low luminance + blue bias → dark wet film
        darkness = np.clip((0.45 - arr.mean(axis=2)) * 2.5, 0.0, 1.0)
        heat = np.clip(glare + darkness * 0.7, 0.0, 1.0)
        try:
            import cv2  # type: ignore[import]
            heat = cv2.GaussianBlur(heat, (15, 15), 0)
        except Exception:  # noqa: BLE001 — blur is purely cosmetic
            pass
        lo, hi = float(heat.min()), float(heat.max())
        return (heat - lo) / (hi - lo + 1e-8)

    @staticmethod
    def _blend_heatmap(image: Image.Image, heat: np.ndarray) -> Image.Image:
        """Upscale the heat grid, colorize with 'jet' and alpha-blend onto frame.

        Args:
            image: Original PIL RGB frame.
            heat: Normalized float32 heat array in [0, 1].

        Returns:
            PIL RGB image with the heatmap overlaid at 45% opacity.
        """
        cmap = _get_cmap("jet")

        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
            image.size, _BICUBIC
        )
        heat_arr = np.asarray(heat_img, dtype=np.float32) / 255.0
        colored = (cmap(heat_arr)[..., :3] * 255).astype(np.uint8)

        base = np.asarray(image.convert("RGB"), dtype=np.float32)
        blended = (0.55 * base + 0.45 * colored.astype(np.float32)).astype(np.uint8)
        return Image.fromarray(blended)

    # ------------------------------------------------------------------
    # Video handling
    # ------------------------------------------------------------------
    def process_video_file(
        self,
        video_path: str,
        sample_interval_sec: float = 1.0,
    ) -> List[Tuple[float, Image.Image, Dict[str, float]]]:
        """Sample frames from a video at regular intervals and classify each.

        Args:
            video_path: Absolute or relative path to a local video file.
            sample_interval_sec: Time in seconds between sampled frames.

        Returns:
            List of ``(timestamp_sec, frame_image, condition_probs)`` tuples,
            ordered chronologically. Returns an empty list if the file cannot
            be read (the caller decides how to surface this).
        """
        try:
            import cv2  # type: ignore[import]
        except ImportError:
            logger.warning("opencv-python-headless not installed; video mode unavailable.")
            return []

        results: List[Tuple[float, Image.Image, Dict[str, float]]] = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Could not open video: %s", video_path)
            return results

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0:
            fps = 25.0
        step = max(1, int(round(fps * sample_interval_sec)))

        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                probs = self.classify_track(pil_frame)
                results.append((round(index / fps, 2), pil_frame, probs))
            index += 1

        cap.release()
        return results
