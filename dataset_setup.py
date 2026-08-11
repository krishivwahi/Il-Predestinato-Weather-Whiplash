"""Upload sample track-condition images to the Hugging Face Dataset Hub.

Automates publishing a local folder of 20–30 sample track frames, organised
as ``dry/``, ``damp/``, ``wet/`` and ``drying/`` subfolders, to a Hugging Face
Dataset repository. If the local folder is missing, synthetic samples are
generated with the project's TrackStreamSimulator so the upload always has
content.

Usage:
    huggingface-cli login          # or set HF_TOKEN env var
    python dataset_setup.py --repo-id your-team/weather-whiplash-track-frames
    python dataset_setup.py --repo-id your-team/weather-whiplash-track-frames \
        --folder ./track_dataset --per-category 7 --private
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

from huggingface_hub import HfApi, create_repo

from simulation import TrackStreamSimulator

CATEGORIES: Dict[str, float] = {
    # category -> representative wetness level for synthetic generation
    "dry": 0.05,
    "drying": 0.35,
    "damp": 0.60,
    "wet": 0.95,
}


def ensure_samples(folder: str, per_category: int) -> None:
    """Generate synthetic category samples for any missing/empty subfolder."""
    simulator = TrackStreamSimulator(seed=11)
    for category, wetness in CATEGORIES.items():
        subdir = os.path.join(folder, category)
        os.makedirs(subdir, exist_ok=True)
        existing = [
            name
            for name in os.listdir(subdir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        if existing:
            print(f"[skip] {subdir} already has {len(existing)} images")
            continue
        for index in range(per_category):
            jitter = (index - per_category / 2) * 0.03
            frame = simulator.frame_for_wetness(wetness + jitter)
            path = os.path.join(subdir, f"{category}_{index + 1:02d}.png")
            frame.save(path)
        print(f"[gen ] wrote {per_category} synthetic frames to {subdir}")


def write_dataset_card(folder: str, repo_id: str) -> None:
    """Write a minimal dataset card so the Hub page renders cleanly."""
    card = (
        "---\n"
        "license: mit\n"
        "task_categories:\n"
        "- image-classification\n"
        "tags:\n"
        "- formula1\n"
        "- track-conditions\n"
        "- weather\n"
        "---\n\n"
        f"# {repo_id.split('/')[-1]}\n\n"
        "Sample track-surface frames for the **Weather Whiplash: Live Track "
        "Condition Detector** hackathon project, organised into four classes: "
        "`dry/`, `damp/`, `wet/`, `drying/`.\n\n"
        "Used to validate zero-shot CLIP classification "
        "(`openai/clip-vit-base-patch32`) of live track conditions.\n"
    )
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(card)


def main() -> None:
    """Parse arguments, prepare the folder, and upload to the Dataset Hub."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target dataset repo, e.g. your-team/weather-whiplash-track-frames",
    )
    parser.add_argument(
        "--folder",
        default="track_dataset",
        help="Local dataset folder with dry/ damp/ wet/ drying/ subfolders",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=7,
        help="Synthetic images per category when a subfolder is empty (7x4=28)",
    )
    parser.add_argument("--private", action="store_true", help="Create a private repo")
    args = parser.parse_args()

    ensure_samples(args.folder, args.per_category)
    write_dataset_card(args.folder, args.repo_id)

    api = HfApi()
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload Weather Whiplash track-condition sample frames",
    )
    print(f"\nDone → https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
