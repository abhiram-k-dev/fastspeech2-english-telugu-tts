"""Telugu preprocessing script.

This script prepares the IndicTTS Telugu dataset for FastSpeech 2 training.

Steps performed:
  1. Fix MFA TextGrid silent-phone labels (empty strings → 'sp').
  2. Drop audio clips that have no matching TextGrid.
  3. Build the Telugu grapheme symbol list from the aligned TextGrids
     and write text/telugu_symbols.json + text/symbols.py.
  4. Extract mel-spectrogram, pitch (F0), energy, and duration features
     using the standard ming024/FastSpeech2 Preprocessor.
  5. Overwrite the random train/val split with the manifest-defined split.

Changes vs. the original ming024 preprocessor:
  - Telugu uses a *grapheme-level* representation (no phoneme dictionary).
    The MFA aligner was trained on a Telugu grapheme model, so the TextGrid
    phones tier contains Unicode Telugu graphemes directly.
  - The grapheme symbol list is built dynamically from the aligned data,
    written to text/telugu_symbols.json, and loaded by text/symbols.py.
  - text/symbols.py is regenerated to load Telugu symbols (not English ARPAbet).

Usage (from the repo root directory):
    python Telugu/preprocess.py Telugu/config.yaml

After running this, text/symbols.py will contain Telugu grapheme symbols.
If you then want to run English inference, you must restore the English symbols.py
(it will be overwritten by this script). Keep separate working copies or use
the config-aware symbol loading approach.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts_helpers as h
from preprocessor.preprocessor import Preprocessor


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess the IndicTTS Telugu dataset for FastSpeech 2."
    )
    ap.add_argument(
        "config",
        type=str,
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to the Telugu config.yaml (default: Telugu/config.yaml)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    # Support both combined (preprocess: ...) and flat YAML.
    if "preprocess" in cfg:
        pre_section = cfg["preprocess"]
    else:
        pre_section = cfg

    p = pre_section["path"]
    pre = p["preprocessed_path"]
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(pre, exist_ok=True)

    print("=" * 60)
    print("Step 1: Fixing MFA TextGrid silence labels ('' → 'sp')...")
    n_fixed = h.fix_textgrids(p["aligned_path"], os.path.join(pre, "TextGrid"))
    print(f"  TextGrids processed: {n_fixed}")

    print("Step 2: Removing clips without a matching TextGrid...")
    removed = h.drop_clips_without_textgrid(p["raw_path"], os.path.join(pre, "TextGrid"))
    print(f"  Clips dropped: {len(removed)}")

    # Telugu-specific: collect grapheme symbols from TextGrid phones tier.
    print("Step 3: Building Telugu grapheme symbol list from TextGrids...")
    phones = h.collect_phones(os.path.join(pre, "TextGrid"))
    chars = h.write_telugu_symbols(phones, code_dir)
    print(f"  Telugu grapheme symbols: {len(chars)}")
    print(f"  Written to: {code_dir}/text/telugu_symbols.json")
    print(f"              {code_dir}/text/symbols.py")

    print("Step 4: Extracting acoustic features (mel, pitch, energy, duration)...")
    Preprocessor(pre_section).build_from_path()

    print("Step 5: Applying manifest-defined train/val split...")
    counts = h.rewrite_split(pre, p["manifest_path"])
    print(f"  Split: {counts}")
    print("=" * 60)
    print("Telugu preprocessing complete.")


if __name__ == "__main__":
    main()
