"""English preprocessing script.

This script prepares the Svarah English dataset for FastSpeech 2 training.

Steps performed:
  1. Fix MFA TextGrid silent-phone labels (empty → 'sp').
  2. Drop audio clips that have no matching TextGrid.
  3. Extract mel-spectrogram, pitch (F0), energy, and duration features
     using the standard ming024/FastSpeech2 Preprocessor.
  4. Overwrite the random train/val split with the manifest-defined split.

Usage (run from the repo root directory, e.g. in Google Colab):
    python English/preprocess.py English/config.yaml

Outputs written to preprocessed_path (from the config):
    train.txt, val.txt  — metadata files for training
    mel/                — mel-spectrogram arrays (.npy)
    pitch/              — pitch arrays (.npy, phoneme-level, normalised)
    energy/             — energy arrays (.npy, phoneme-level, normalised)
    duration/           — duration arrays (.npy, in mel frames)
    speakers.json       — speaker-name-to-id mapping
    stats.json          — pitch/energy mean and std

Changes vs. the original ming024 preprocessor:
  - Uses MFA TextGrids (Svarah was aligned with the Montreal Forced Aligner
    using the en_us_arpa acoustic model and the CMU + Svarah dictionary).
  - Empty silence labels in TextGrids are renamed to 'sp' before processing.
  - The final train/val split is taken from the manifest CSV, not random.
"""
import argparse
import os
import sys

import yaml

# Allow running from either the repo root or the English/ subdirectory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts_helpers as h
from preprocessor.preprocessor import Preprocessor


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess the Svarah English dataset for FastSpeech 2."
    )
    ap.add_argument(
        "config",
        type=str,
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to the English config.yaml (default: English/config.yaml)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    p = cfg["preprocess"]["path"] if "preprocess" in cfg else cfg["path"]
    pre_cfg = cfg["preprocess"]["preprocessing"] if "preprocess" in cfg else cfg["preprocessing"]

    # Flatten config to what Preprocessor expects (it takes just the preprocess section).
    if "preprocess" in cfg:
        pre_section = cfg["preprocess"]
    else:
        pre_section = cfg

    pre = p["preprocessed_path"]
    os.makedirs(pre, exist_ok=True)

    print("=" * 60)
    print("Step 1: Fixing MFA TextGrid silence labels...")
    n_fixed = h.fix_textgrids(p["aligned_path"], os.path.join(pre, "TextGrid"))
    print(f"  TextGrids processed: {n_fixed}")

    print("Step 2: Removing clips without a TextGrid...")
    removed = h.drop_clips_without_textgrid(p["raw_path"], os.path.join(pre, "TextGrid"))
    print(f"  Clips dropped: {len(removed)}")
    if removed:
        print(f"  First few: {removed[:5]}")

    # English uses ARPAbet phonemes (from MFA) — no symbol file regeneration needed.
    # The symbols.py for English must point to the ARPAbet symbol set.
    print("Step 3: Extracting acoustic features (mel, pitch, energy, duration)...")
    Preprocessor(pre_section).build_from_path()

    print("Step 4: Applying manifest-defined train/val split...")
    counts = h.rewrite_split(pre, p["manifest_path"])
    print(f"  Split: {counts}")
    print("=" * 60)
    print("English preprocessing complete.")


if __name__ == "__main__":
    main()
