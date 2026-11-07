"""Language-agnostic preprocessing wrapper around the ming024 Preprocessor.
Usage: python preprocess.py config/preprocess.yaml
Steps: (1) copy MFA TextGrids with empty silences renamed to 'sp'; (2) drop clips without a TextGrid;
(3) Telugu only: build the grapheme symbol list; (4) run the repo's feature extraction (mel, pitch, energy, duration);
(5) overwrite the repo's random train/val split with our own manifest split."""
import argparse, os
import yaml
import tts_helpers as h
from preprocessor.preprocessor import Preprocessor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=str, help="path to preprocess.yaml")
    cfg = yaml.load(open(ap.parse_args().config, encoding="utf-8"), Loader=yaml.FullLoader)
    p = cfg["path"]
    pre = p["preprocessed_path"]
    os.makedirs(pre, exist_ok=True)
    print("TextGrids fixed:", h.fix_textgrids(p["aligned_path"], f"{pre}/TextGrid"))
    removed = h.drop_clips_without_textgrid(p["raw_path"], f"{pre}/TextGrid")
    print("clips dropped (no TextGrid):", len(removed))
    if cfg["preprocessing"]["text"]["language"] == "te":
        chars = h.write_telugu_symbols(h.collect_phones(f"{pre}/TextGrid"), os.path.dirname(os.path.abspath(__file__)))
        print("Telugu symbols:", len(chars))
    Preprocessor(cfg).build_from_path()
    print("split (from manifest):", h.rewrite_split(pre, p["manifest_path"]))
