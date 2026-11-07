"""English inference. Usage (from the code folder):
python inference_english.py --restore_step 10000 -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml --out Results/English
Generates wav + duration/pitch/energy plots for E1, E2a, E2b, E3, E4 (text preserved as given)."""
import argparse, collections, json, os, sys
import torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_helpers as h
from utils.model import get_model, get_vocoder

TEST = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
}
# Heteronym "conduct": noun = stress on the first syllable, verb = stress on the second syllable.
OVERRIDES = {
    "E2a": {"conduct": "K AA1 N D AH0 K T".split()},
    "E2b": {"conduct": "K AH0 N D AH1 K T".split()},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument("-p", "--preprocess_config", required=True)
    ap.add_argument("-m", "--model_config", required=True)
    ap.add_argument("-t", "--train_config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    configs = [yaml.safe_load(open(p, encoding="utf-8")) for p in
               [args.preprocess_config, args.model_config, args.train_config]]
    preprocess_config, model_config, train_config = configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    lex = h.read_lexicon(preprocess_config["path"]["lexicon_path"])
    from g2p_en import G2p
    g2p = G2p()
    for name, text in TEST.items():
        ph = h.en_phonemes(text, lex, g2p, OVERRIDES.get(name))
        print(name, "|", text, "|", ph)
        h.synthesize_and_plot(ph, name, text, 0, model, vocoder, configs, device,
                              f"{args.out}/wavs", args.out)

if __name__ == "__main__":
    main()
