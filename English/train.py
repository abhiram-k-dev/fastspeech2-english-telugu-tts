"""English training script.

Trains the FastSpeech 2 model on the Svarah English dataset.

The model is a multi-speaker FastSpeech 2 with:
  - 4 encoder FFT blocks (2 heads, 256-dim hidden)
  - 6 decoder FFT blocks (2 heads, 256-dim hidden)
  - Phoneme-level pitch and energy predictors
  - Duration predictor
  - HiFi-GAN universal vocoder for waveform synthesis during training

Before running this script:
  1. Run English/preprocess.py to build the preprocessed dataset.
  2. Download the HiFi-GAN universal vocoder checkpoint:
       hifigan/generator_universal.pth.tar
       hifigan/config.json
     (available at https://github.com/jik876/hifi-gan)
  3. Optionally apply compatibility patches with tts_helpers.patch_repo().

Usage (from the repo root directory):
    # Start training from scratch:
    python English/train.py \\
        -p English/config.yaml -m English/config.yaml -t English/config.yaml

    # Resume from checkpoint step 5000:
    python English/train.py --restore_step 5000 \\
        -p English/config.yaml -m English/config.yaml -t English/config.yaml

Note: config.yaml contains all three sections (preprocess, model, train).
      The standard train.py expects three separate YAML files.
      Use make_configs.py (or split_config in tts_helpers) to split English/config.yaml
      into config/preprocess.yaml, config/model.yaml, config/train.yaml before training.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts_helpers as h

# Apply compatibility patches for modern librosa/torch before importing model code.
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
h.patch_repo(_CODE_DIR)

from train import main as _train_main


def _load_split_config(config_path):
    """Load a combined config.yaml and return (preprocess, model, train) dicts."""
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    return cfg["preprocess"], cfg["model"], cfg["train"]


def main():
    ap = argparse.ArgumentParser(
        description="Train FastSpeech 2 on the Svarah English dataset."
    )
    ap.add_argument("--restore_step", type=int, default=0,
                    help="Resume training from this checkpoint step (0 = fresh start).")
    ap.add_argument(
        "-p", "--preprocess_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to preprocess config (or combined config.yaml).",
    )
    ap.add_argument(
        "-m", "--model_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to model config (or combined config.yaml).",
    )
    ap.add_argument(
        "-t", "--train_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to train config (or combined config.yaml).",
    )
    args = ap.parse_args()

    # Support combined config.yaml or separate three-file style.
    def _load(path):
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        return cfg

    pc = _load(args.preprocess_config)
    mc = _load(args.model_config)
    tc = _load(args.train_config)

    # If the loaded file has all three sections, extract them.
    if "preprocess" in pc:
        pc = pc["preprocess"]
    if "model" in mc:
        mc = mc["model"]
    if "train" in tc:
        tc = tc["train"]

    configs = (pc, mc, tc)
    _train_main(args, configs)


if __name__ == "__main__":
    main()
