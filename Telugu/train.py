"""Telugu training script.

Trains the FastSpeech 2 model on the IndicTTS Telugu dataset.

The model is a single-speaker FastSpeech 2 with:
  - 4 encoder FFT blocks (2 heads, 256-dim hidden)
  - 6 decoder FFT blocks (2 heads, 256-dim hidden)
  - Phoneme(grapheme)-level pitch and energy predictors
  - Duration predictor
  - HiFi-GAN universal vocoder for waveform synthesis during training

Before running this script:
  1. Run Telugu/preprocess.py to build the preprocessed dataset.
     This also regenerates text/symbols.py for Telugu.
  2. Download the HiFi-GAN universal vocoder checkpoint:
       hifigan/generator_universal.pth.tar
       hifigan/config.json
     (available at https://github.com/jik876/hifi-gan)

Usage (from the repo root directory):
    # Start training from scratch:
    python Telugu/train.py \\
        -p Telugu/config.yaml -m Telugu/config.yaml -t Telugu/config.yaml

    # Resume from checkpoint step 1000:
    python Telugu/train.py --restore_step 1000 \\
        -p Telugu/config.yaml -m Telugu/config.yaml -t Telugu/config.yaml

Changes vs. English training:
  - multi_speaker: false (IndicTTS Telugu is single-speaker)
  - Symbol set: Telugu Unicode graphemes (63 characters) instead of ARPAbet
  - Text normalisation: Telugu number/ordinal words instead of English
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts_helpers as h

# Apply compatibility patches before importing model code.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
h.patch_repo(_CODE_DIR)

from train import main as _train_main


def main():
    ap = argparse.ArgumentParser(
        description="Train FastSpeech 2 on the IndicTTS Telugu dataset."
    )
    ap.add_argument("--restore_step", type=int, default=0,
                    help="Resume training from this checkpoint step (0 = fresh start).")
    ap.add_argument(
        "-p", "--preprocess_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    ap.add_argument(
        "-m", "--model_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    ap.add_argument(
        "-t", "--train_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    args = ap.parse_args()

    def _load(path, section):
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        return cfg[section] if section in cfg else cfg

    pc = _load(args.preprocess_config, "preprocess")
    mc = _load(args.model_config, "model")
    tc = _load(args.train_config, "train")
    configs = (pc, mc, tc)
    _train_main(args, configs)


if __name__ == "__main__":
    main()
