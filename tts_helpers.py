"""Shared helpers for the FastSpeech 2 English + Telugu project (patching, data prep, text front-ends, plots)."""
import os, re, glob, json, unicodedata
import numpy as np

# ----------------------------------------------------------------------------
# 1. Patch the ming024/FastSpeech2 code so it runs with modern librosa / torch
# ----------------------------------------------------------------------------
def _sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    if new in s and old not in s:
        print("already patched:", path, "|", old[:40].replace("\n", " "))
        return
    assert s.count(old) == count, f"{path}: expected {count} match(es) of {old[:60]!r}, found {s.count(old)}"
    open(path, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
    print("patched:", path, "|", old[:40].replace("\n", " "))

def patch_repo(code_dir):
    _sub(f"{code_dir}/audio/stft.py", "pad_center(fft_window, filter_length)", "pad_center(fft_window, size=filter_length)")
    _sub(f"{code_dir}/audio/stft.py",
         "librosa_mel_fn(\n            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax\n        )",
         "librosa_mel_fn(\n            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax\n        )")
    _sub(f"{code_dir}/utils/model.py", "torch.load(ckpt_path)", "torch.load(ckpt_path, weights_only=False)")
    _sub(f"{code_dir}/utils/model.py", 'torch.load("hifigan/generator_LJSpeech.pth.tar")',
         'torch.load("hifigan/generator_LJSpeech.pth.tar", weights_only=False)')
    _sub(f"{code_dir}/utils/model.py", 'torch.load("hifigan/generator_universal.pth.tar")',
         'torch.load("hifigan/generator_universal.pth.tar", weights_only=False)')

# ----------------------------------------------------------------------------
# 2. MFA TextGrids: fix empty silence labels and collect Telugu symbols
# ----------------------------------------------------------------------------
def fix_textgrids(tg_root, out_root):
    """Copy TextGrids and rename empty phone labels to 'sp'."""
    import tgt
    n = 0
    for p in glob.glob(f"{tg_root}/**/*.TextGrid", recursive=True):
        spk = os.path.basename(os.path.dirname(p))
        tg = tgt.io.read_textgrid(p)
        tier = tg.get_tier_by_name("phones")
        for iv in tier._objects:
            if iv.text.strip() == "":
                iv.text = "sp"
        os.makedirs(f"{out_root}/{spk}", exist_ok=True)
        tgt.io.write_to_file(tg, f"{out_root}/{spk}/{os.path.basename(p)}", format="long")
        n += 1
    return n

def collect_phones(tg_root):
    """Collect all phone labels from the phones tier of every TextGrid."""
    import tgt
    s = set()
    for p in glob.glob(f"{tg_root}/**/*.TextGrid", recursive=True):
        for iv in tgt.io.read_textgrid(p).get_tier_by_name("phones")._objects:
            if iv.text.strip():
                s.add(iv.text)
    return s

def drop_clips_without_textgrid(raw_dir, tg_root):
    """Remove audio and label files that do not have matching TextGrids."""
    removed = []
    for wav in glob.glob(f"{raw_dir}/*/*.wav"):
        spk, base = os.path.basename(os.path.dirname(wav)), os.path.basename(wav)[:-4]
        if not os.path.exists(f"{tg_root}/{spk}/{base}.TextGrid"):
            for ext in (".wav", ".lab"):
                f = f"{raw_dir}/{spk}/{base}{ext}"
                if os.path.exists(f):
                    os.remove(f)
            removed.append(base)
    return removed

def write_telugu_symbols(phones, code_dir):
    """Create the Telugu grapheme symbol list and symbols.py file."""
    chars = sorted(p for p in phones if p not in ("sp", "spn", "sil"))
    json.dump(chars, open(f"{code_dir}/text/telugu_symbols.json", "w", encoding="utf-8"), ensure_ascii=False)
    open(f"{code_dir}/text/symbols.py", "w", encoding="utf-8").write(
        '"""Telugu grapheme symbols (generated from the MFA phones tier)."""\n'
        "import json, os\n"
        '_pad = "_"\n'
        '_te = json.load(open(os.path.join(os.path.dirname(__file__), "telugu_symbols.json"), encoding="utf-8"))\n'
        'symbols = [_pad] + ["@" + c for c in _te] + ["@sp", "@spn", "@sil"]\n')
    return chars

# ----------------------------------------------------------------------------
# 3. Replace the repository's random train/validation split with our manifest
# ----------------------------------------------------------------------------
def rewrite_split(pre_dir, manifest_csv):
    """Rewrite train.txt and val.txt using the split column in our manifest."""
    import pandas as pd
    m = pd.read_csv(manifest_csv)
    split_of = dict(zip(m["utt"], m["split"]))
    lines = []
    for f in ("train.txt", "val.txt"):
        lines += [l for l in open(f"{pre_dir}/{f}", encoding="utf-8").read().split("\n") if l.strip()]
    out = {"train": [], "val": []}
    for l in lines:
        sp = split_of.get(l.split("|")[0])
        if sp in out:
            out[sp].append(l)
    for k, v in out.items():
        open(f"{pre_dir}/{k}.txt", "w", encoding="utf-8").write("\n".join(v) + "\n")
    return {k: len(v) for k, v in out.items()}

# ----------------------------------------------------------------------------
# 4. English text normalisation and phoneme conversion
# ----------------------------------------------------------------------------
_MONTHS = "january february march april may june july august september october november december".split()

def normalize_en(text):
    """Convert numbers, years and times into spoken English words."""
    import inflect
    eng = inflect.engine()
    words = lambda n: eng.number_to_words(int(n), andword="").replace(",", "").replace("-", " ")

    def year(y):
        y = int(y)
        if 2000 <= y <= 2009:
            return words(y)
        if 1100 <= y <= 2099:
            a, b = divmod(y, 100)
            return words(a) + (" " + ("oh " + words(b) if 0 < b < 10 else words(b)) if b else " hundred")
        return words(y)

    t = text
    def clock(m):
        h, mi, ap = m.group(1), m.group(2), m.group(3).lower()
        out = words(h)
        if mi != "00":
            out += " " + ("oh " + words(mi) if mi[0] == "0" else words(mi))
        return out + " " + ap + " m"
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]\.?", clock, t)
    t = re.sub(r"\b(\d{4})\b", lambda m: year(m.group(1)), t)
    t = re.sub(r"\d+", lambda m: words(m.group(0)), t)
    t = t.replace("&", " and ").replace("%", " percent ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def read_lexicon(path):
    """Load the merged pronunciation dictionary."""
    lex = {}
    for line in open(path, encoding="utf-8"):
        parts = re.split(r"\s+", line.strip("\n"))
        if len(parts) > 1 and parts[0].lower() not in lex:
            lex[parts[0].lower()] = parts[1:]
    return lex

def en_phonemes(text, lexicon, g2p, overrides=None):
    """Convert English text into ARPAbet phonemes, using the lexicon and G2P."""
    overrides = overrides or {}
    t = normalize_en(text).lower()
    t = re.sub(r"[.!?]+$", "", t)
    phones = []
    for w in re.split(r"([,;:\-\?\!\s]+)", t):
        if not w or not w.strip():
            continue
        if re.fullmatch(r"[,;:\-\?\!\s]+", w):
            phones.append("sp")
        elif w in overrides:
            phones += overrides[w]
        elif w in lexicon:
            phones += lexicon[w]
        else:
            phones += [p for p in g2p(w) if p.strip() and re.match(r"^[A-Z]+[0-2]?$", p)]
    return "{" + " ".join(phones) + "}"

# ----------------------------------------------------------------------------
# 5. Telugu text normalisation and grapheme conversion
# ----------------------------------------------------------------------------
_TE_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది",
             "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
_TE_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}

def te_number(n):
    """Convert an integer into Telugu number words."""
    n = int(n)
    if n < 20:
        return _TE_UNITS[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _TE_TENS[t] + ("" if u == 0 else " " + _TE_UNITS[u])
    if n < 1000:
        h, r = divmod(n, 100)
        head = "వంద" if h == 1 else _TE_UNITS[h] + " వందల"
        return head + ("" if r == 0 else " " + te_number(r))
    th, r = divmod(n, 1000)
    head = "వెయ్యి" if th == 1 else te_number(th) + " వేల"
    if r == 0 and th != 1:
        head = te_number(th) + " వేలు"
    return head + ("" if r == 0 else " " + te_number(r))

def normalize_te(text):
    """Convert digits, times and ordinals into Telugu words and normalise Unicode."""
    t = unicodedata.normalize("NFC", text).replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    def tm(m):
        h, mi = int(m.group(1)), int(m.group(2))
        return te_number(h) + ("" if mi == 0 else " " + te_number(mi))
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", tm, t)
    def ordn(m):
        n = int(m.group(1))
        w = "మొదటి" if n == 1 else te_number(n)
        w = w[:-1] + "వ" if (w.endswith("ు") or w.endswith("ి")) and n != 1 else (w + "వ" if n != 1 else w)
        return w
    t = re.sub(r"(\d+)వ\b", ordn, t)
    t = re.sub(r"\d+", lambda m: te_number(m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()

def te_phonemes(text, symbols):
    """Convert normalised Telugu text into grapheme symbols for the model."""
    t = normalize_te(text)
    t = re.sub(r"[.!?]+$", "", t)
    ph, missing = [], set()
    for ch in t:
        if ch.isspace():
            continue
        if ch in ",;:":
            ph.append("sp")
        elif ch in symbols:
            ph.append(ch)
        else:
            missing.add(ch)
    if missing:
        print("WARNING: symbols not seen in training were dropped:", missing)
    return "{" + " ".join(ph) + "}"

# ----------------------------------------------------------------------------
# 6. Inference and variance-adaptor plots
# ----------------------------------------------------------------------------
def find_telugu_font():
    """Find an installed font that can display Telugu characters."""
    from matplotlib import font_manager as fm
    for p in glob.glob("/usr/share/fonts/**/*Telugu*", recursive=True) + glob.glob("/usr/share/fonts/**/*telu*", recursive=True):
        try:
            fm.fontManager.addfont(p)
            return fm.FontProperties(fname=p).get_name()
        except Exception:
            continue
    return None

def plot_variance(sym, dur_frames, pitch_hz, energy, out_dir, name, title, font=None, hop=256, sr=22050):
    """Save duration, pitch and energy plots, together with a CSV containing their values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(sym))
    labels = [s if font else str(i) for i, s in enumerate(sym)]
    series = {
        "duration": (dur_frames * hop / sr * 1000.0, "Predicted duration (ms)", "bar"),
        "pitch": (pitch_hz, "Predicted pitch F0 (Hz)", "line"),
        "energy": (energy, "Predicted energy", "line"),
    }
    for key, (vals, ylab, kind) in series.items():
        os.makedirs(f"{out_dir}/{key}", exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(sym)), 3.6))
        if font:
            plt.rcParams["font.family"] = [font, "DejaVu Sans"]
        if kind == "bar":
            ax.bar(x, vals)
        else:
            ax.plot(x, vals, marker="o", ms=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_xlabel("Phoneme / grapheme index" if not font else "Phoneme / grapheme")
        ax.set_ylabel(ylab)
        ax.set_title(f"{name}: {title}", fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{key}/{name}_{key}.png", dpi=150)
        plt.close(fig)
    import csv
    os.makedirs(f"{out_dir}/csv", exist_ok=True)
    with open(f"{out_dir}/csv/{name}_variance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "symbol", "duration_frames", "duration_ms", "pitch_hz", "energy"])
        for i, s in enumerate(sym):
            w.writerow([i, s, int(dur_frames[i]), round(float(dur_frames[i]) * hop / sr * 1000, 1), round(float(pitch_hz[i]), 2), round(float(energy[i]), 4)])

def synthesize_and_plot(phone_string, name, title, speaker_id, model, vocoder, configs, device, wav_dir, plot_dir, font=None):
    """Generate a WAV file and its duration, pitch and energy plots."""
    import torch
    from scipy.io import wavfile
    from text import text_to_sequence
    from utils.model import vocoder_infer
    preprocess_config, model_config, train_config = configs
    seq = np.array(text_to_sequence(phone_string, preprocess_config["preprocessing"]["text"]["text_cleaners"]))
    sym = phone_string.strip("{}").split()
    assert len(seq) == len(sym), f"{len(sym)} symbols but {len(seq)} ids: some symbols are not in the symbol list"
    src = torch.from_numpy(seq).long().unsqueeze(0).to(device)
    src_lens = torch.tensor([len(seq)]).to(device)
    spk = torch.tensor([speaker_id]).long().to(device)
    with torch.no_grad():
        out = model(spk, src, src_lens, int(src_lens.max()))
        post, p_pred, e_pred, logd, d_round, _, _, _, mel_len = out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9]
        wav = vocoder_infer(post.transpose(1, 2), vocoder, model_config, preprocess_config,
                            lengths=mel_len * preprocess_config["preprocessing"]["stft"]["hop_length"])[0]
    st = json.load(open(os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")))
    p_mean, p_std = st["pitch"][2], st["pitch"][3]
    e_mean, e_std = st["energy"][2], st["energy"][3]
    n = len(seq)
    pitch = p_pred[0, :n].cpu().numpy() * p_std + p_mean if preprocess_config["preprocessing"]["pitch"]["normalization"] else p_pred[0, :n].cpu().numpy()
    energy = e_pred[0, :n].cpu().numpy() * e_std + e_mean if preprocess_config["preprocessing"]["energy"]["normalization"] else e_pred[0, :n].cpu().numpy()
    dur = d_round[0, :n].cpu().numpy()
    os.makedirs(wav_dir, exist_ok=True)
    wavfile.write(f"{wav_dir}/{name}.wav", preprocess_config["preprocessing"]["audio"]["sampling_rate"], wav)
    plot_variance(sym, dur, pitch, energy, plot_dir, name, title, font=font,
                  hop=preprocess_config["preprocessing"]["stft"]["hop_length"],
                  sr=preprocess_config["preprocessing"]["audio"]["sampling_rate"])
    return sym, dur, pitch, energy

# ----------------------------------------------------------------------------
# 7. Parse training logs and generate loss curves
# ----------------------------------------------------------------------------
def parse_log(path):
    """Read total and component losses from a FastSpeech 2 log file."""
    rows = []
    pat = re.compile(r"Step (\d+)(?:/\d+)?, Total Loss: ([\d.]+), Mel Loss: ([\d.]+), Mel PostNet Loss: ([\d.]+), Pitch Loss: ([\d.]+), Energy Loss: ([\d.]+), Duration Loss: ([\d.]+)")
    for line in open(path, encoding="utf-8"):
        m = pat.search(line)
        if m:
            rows.append([int(m.group(1))] + [float(m.group(i)) for i in range(2, 8)])
    return np.array(rows)

def plot_losses(log_root, out_dir, language):
    """Save separate training and validation loss plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    names = ["Total", "Mel", "Mel PostNet", "Pitch", "Energy", "Duration"]
    for split, fname in [("train", "training_loss.png"), ("val", "validation_loss.png")]:
        a = parse_log(f"{log_root}/{split}/log.txt")
        if len(a) == 0:
            print("no rows parsed for", split)
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13, 6))
        for i, ax in enumerate(axes.ravel()):
            ax.plot(a[:, 0], a[:, 1 + i])
            ax.set_title(f"{names[i]} loss")
            ax.set_xlabel("step")
        fig.suptitle(f"{language}: {'training' if split == 'train' else 'validation'} loss")
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{fname}", dpi=150)
        plt.close(fig)

# ----------------------------------------------------------------------------
# 8. Build the combined configuration for each language
# ----------------------------------------------------------------------------
def build_config(lang, root, raw_dir, aligned_dir, manifest_csv, total_step=10000, batch_size=16):
    """Build the language-specific FastSpeech 2 configuration dictionary."""
    en = lang == "english"
    name = "English" if en else "Telugu"
    pre = f"/content/preprocessed_data/{name}"
    return {
        "preprocess": {
            "dataset": "Svarah_English" if en else "IndicTTS_Telugu",
            "path": {
                "corpus_path": "", "raw_path": raw_dir, "preprocessed_path": pre,
                "lexicon_path": f"{root}/processed/english/en_merged.dict" if en else "",
                "aligned_path": aligned_dir, "manifest_path": manifest_csv,
            },
            "preprocessing": {
                "val_size": 512,
                "text": {"text_cleaners": ["basic_cleaners"], "language": "en" if en else "te"},
                "audio": {"sampling_rate": 22050, "max_wav_value": 32768.0},
                "stft": {"filter_length": 1024, "hop_length": 256, "win_length": 1024},
                "mel": {"n_mel_channels": 80, "mel_fmin": 0, "mel_fmax": 8000},
                "pitch": {"feature": "phoneme_level", "normalization": True},
                "energy": {"feature": "phoneme_level", "normalization": True},
            },
        },
        "model": {
            "transformer": {"encoder_layer": 4, "encoder_head": 2, "encoder_hidden": 256, "decoder_layer": 6,
                            "decoder_head": 2, "decoder_hidden": 256, "conv_filter_size": 1024,
                            "conv_kernel_size": [9, 1], "encoder_dropout": 0.2, "decoder_dropout": 0.2},
            "variance_predictor": {"filter_size": 256, "kernel_size": 3, "dropout": 0.5},
            "variance_embedding": {"pitch_quantization": "linear", "energy_quantization": "linear", "n_bins": 256},
            "multi_speaker": bool(en),
            "max_seq_len": 1000,
            "vocoder": {"model": "HiFi-GAN", "speaker": "universal"},
        },
        "train": {
            "path": {"ckpt_path": f"{root}/checkpoints/{lang}", "log_path": f"{root}/results/{lang}/log",
                     "result_path": f"{root}/results/{lang}/synth"},
            "optimizer": {"batch_size": batch_size, "betas": [0.9, 0.98], "eps": 1e-9, "weight_decay": 0.0,
                          "grad_clip_thresh": 1.0, "grad_acc_step": 1, "warm_up_step": 1000,
                          "anneal_steps": [int(total_step * 0.6), int(total_step * 0.8), int(total_step * 0.9)],
                          "anneal_rate": 0.3},
            "step": {"total_step": total_step, "log_step": 50, "synth_step": 1000, "val_step": 500, "save_step": 1000},
        },
    }

def split_config(path="config.yaml", out_dir="config"):
    """Split the combined configuration into the three YAML files expected by the repo."""
    import yaml
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)
    for k in ("preprocess", "model", "train"):
        yaml.safe_dump(cfg[k], open(f"{out_dir}/{k}.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    return list(cfg)
