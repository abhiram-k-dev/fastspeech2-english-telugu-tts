"""English inference script.

Generates speech and variance-adaptor analysis plots for 5 English test sentences
using the trained FastSpeech 2 model.

Test sentences (chosen to test specific linguistic phenomena):
  E1: Long polysyllabic word — "antidisestablishmentarianism"
  E2a/E2b: Heteronym — "conduct" (noun vs. verb, different stress patterns)
  E3: Intentional typo — "rhyth m" (space in word, to test robustness)
  E4: Mixed content — digits, AM/PM time, proper nouns

Outputs per sentence (saved to --out directory):
  wavs/<ID>.wav                             — synthesised speech
  <ID>_phonemes.txt                         — phoneme sequence used
  variance_adaptor_plots/duration/<ID>_duration.png  — duration bar chart
  variance_adaptor_plots/pitch/<ID>_pitch.png        — pitch F0 line plot
  variance_adaptor_plots/energy/<ID>_energy.png      — energy line plot
  variance_adaptor_plots/csv/<ID>_variance.csv       — numeric data

Changes vs. the original ming024 synthesize.py for English:
  - Standalone inference without needing a source text file.
  - Number/date/time normalisation before G2P (via tts_helpers.normalize_en).
  - Heteronym override support (phoneme dict override per word).
  - Variance adaptor analysis plots saved automatically.

Usage (from the repo root):
    python English/inference.py \\
        --restore_step 10000 \\
        -p English/config.yaml \\
        -m English/config.yaml \\
        -t English/config.yaml \\
        --out Results/English

Prerequisites:
  - Trained checkpoint at <ckpt_path>/<restore_step>.pth.tar
  - HiFi-GAN universal vocoder checkpoint at hifigan/generator_universal.pth.tar
  - Preprocessed data at <preprocessed_path> (for stats.json + speakers.json)
"""
import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts_helpers as h
from utils.model import get_model, get_vocoder

# ---------------------------------------------------------------------------
# 5 test sentences (preserving text exactly as given in the assignment)
# ---------------------------------------------------------------------------
TEST = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum "
          "during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, "
          "so andaroo should arrive on time.",
}

# Heteronym overrides: "conduct" pronounced differently as noun vs. verb.
#   E2a: noun  → stress on 1st syllable: KAA1-N D AH0 K T
#   E2b: verb  → stress on 2nd syllable: K AH0 N D AH1 K T
OVERRIDES = {
    "E2a": {"conduct": "K AA1 N D AH0 K T".split()},
    "E2b": {"conduct": "K AH0 N D AH1 K T".split()},
}


def _load_section(path, section):
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    return cfg[section] if section in cfg else cfg


def main():
    ap = argparse.ArgumentParser(
        description="Run English FastSpeech 2 inference and generate variance analysis plots."
    )
    ap.add_argument("--restore_step", type=int, required=True,
                    help="Checkpoint step to load (e.g. 10000).")
    ap.add_argument("-p", "--preprocess_config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("-m", "--model_config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("-t", "--train_config", type=str,
                    default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--out", type=str, default="Results/English",
                    help="Output directory for WAV files and plots.")
    ap.add_argument("--speaker_id", type=int, default=0,
                    help="Speaker ID to use for synthesis (default: 0).")
    args = ap.parse_args()

    pc = _load_section(args.preprocess_config, "preprocess")
    mc = _load_section(args.model_config, "model")
    tc = _load_section(args.train_config, "train")
    configs = (pc, mc, tc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(mc, device)

    lex = h.read_lexicon(pc["path"]["lexicon_path"])
    from g2p_en import G2p
    g2p = G2p()

    wav_dir = os.path.join(args.out, "test_generated_audios")
    plot_dir = os.path.join(args.out, "variance_adaptor_plots")

    for name, text in TEST.items():
        ph = h.en_phonemes(text, lex, g2p, OVERRIDES.get(name))
        print(f"\n{name} | {text}")
        print(f"      Phonemes: {ph}")

        h.synthesize_and_plot(
            ph, name, text, args.speaker_id,
            model, vocoder, configs, device,
            wav_dir, plot_dir,
        )
        # Save phoneme sequence for reference
        ph_file = os.path.join(wav_dir, f"{name}_phonemes.txt")
        os.makedirs(wav_dir, exist_ok=True)
        open(ph_file, "w", encoding="utf-8").write(ph)
        print(f"      Saved: {wav_dir}/{name}.wav + plots")

    print(f"\nAll outputs written to: {args.out}")


if __name__ == "__main__":
    main()
