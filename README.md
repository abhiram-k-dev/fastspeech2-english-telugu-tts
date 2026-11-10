# FastSpeech 2 for English and Telugu Text-to-Speech

**Author:** Abhiram K  
**Reference Paper:** [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)  
**Reference Implementation:** [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)

---

## 1. Project Overview

This project implements FastSpeech 2 Text-to-Speech (TTS) for two languages:
- **English** (Indian-accented) using the [Svarah dataset](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Telugu** using the [IndicTTS Telugu dataset](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)

The codebase is adapted from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) with language-specific changes for Telugu support, documented in the technical report (`Report/report.md`).

---

## 2. Model Architecture

FastSpeech 2 is a **non-autoregressive** parallel neural TTS model:

```
Input text
  → Text normalisation
  → Phoneme/grapheme conversion
  → Symbol-to-ID embedding

[Encoder]       4 × FFT blocks (2 heads, 256-dim hidden)
[Variance Adaptor]
  ├─ Duration Predictor → length regulator
  ├─ Pitch Predictor    → phoneme-level F0
  └─ Energy Predictor   → phoneme-level energy
[Decoder]       6 × FFT blocks (2 heads, 256-dim hidden)
[Linear]        → 80-band mel-spectrogram
[PostNet]       → refined mel-spectrogram
[HiFi-GAN]      → 22050 Hz waveform
```

See `Report/report.md` for the complete architecture description.

---

## 3. English vs. Telugu — Key Differences

| Property | English | Telugu |
|----------|---------|--------|
| Dataset | Svarah (Indian-accented) | IndicTTS Telugu |
| Multi-speaker | Yes | No (single speaker) |
| Phone set | ARPAbet (via g2p_en + dictionary) | Telugu Unicode graphemes (63 symbols) |
| Lexicon | CMU + Svarah dictionary | None (grapheme-to-grapheme) |
| Text normalisation | English number/date/time words | Telugu number/ordinal words |
| Symbol file | Standard ARPAbet symbols | `text/telugu_symbols.json` |

---

## 4. Repository Structure

```
English/              English-specific scripts and config
  config.yaml         Combined preprocess + model + train config (English)
  preprocess.py       English preprocessing entry point
  train.py            English training entry point
  inference.py        English inference (5 test sentences, E1–E4)
  checkpoints/        Trained model checkpoints

Telugu/               Telugu-specific scripts and config
  config.yaml         Combined preprocess + model + train config (Telugu)
  preprocess.py       Telugu preprocessing (builds grapheme symbol list)
  train.py            Telugu training entry point
  inference.py        Telugu inference (5 test sentences, T1–T4)
  checkpoints/        Trained model checkpoints

audio/                Audio processing and STFT
config/               Shared config files (preprocess.yaml, model.yaml, train.yaml)
hifigan/              HiFi-GAN vocoder implementation and config
model/                FastSpeech 2 model (fastspeech2.py, modules.py, loss.py)
preprocessor/         Dataset preprocessing (MFA TextGrid → features)
text/                 Text processing, symbols, Telugu grapheme symbols
transformer/          FFT block implementation
utils/                Utility functions (model loading, tools)

Results/
  English/
    test_generated_audios/    5 English WAV samples (E1–E4)
    variance_adaptor_plots/   Duration, pitch, energy plots (15 total)
  Telugu/
    test_generated_audios/    Telugu WAV samples (after training)
    variance_adaptor_plots/   Telugu variance plots

Report/
  report.md           Technical report (architecture, datasets, results)

tts_helpers.py        Shared utilities (patches, text front-ends, plots)
preprocess.py         Language-agnostic preprocessing wrapper
train.py              Training loop
synthesize.py         Single-sentence and batch synthesis
inference_english.py  English-specific inference script (top-level)
inference_telugu.py   Telugu-specific inference script (top-level)
make_configs.py       Split combined config.yaml into 3 YAML files
config.yaml           Combined English config
telugu_config.yaml    Combined Telugu config
requirements.txt      Python dependencies
```

---

## 5. Installation

```bash
# Clone
git clone https://github.com/abhiram-k-dev/fastspeech2-english-telugu-tts.git
cd fastspeech2-english-telugu-tts

# Install dependencies (Python 3.8–3.10, GPU recommended)
pip install -r requirements.txt

# Download HiFi-GAN universal vocoder checkpoint
# Place generator_universal.pth.tar and config.json in hifigan/
# Available from https://github.com/jik876/hifi-gan

# Optional: install Telugu font for readable plot labels (Ubuntu/Colab)
# sudo apt-get install -y fonts-telu
```

---

## 6. Data Preparation

Both datasets require **forced alignment** with MFA before preprocessing.

### 6.1 English (Svarah)

```bash
# Install MFA
pip install montreal-forced-aligner

# Align (using the en_us_arpa acoustic model and combined dictionary)
mfa align \
    raw_en/ \
    en_merged.dict \
    english_us_arpa \
    aligned_en/
```

### 6.2 Telugu (IndicTTS)

Telugu alignment uses a grapheme model (each Unicode character is a "phone"):

```bash
# Align using the ipa acoustic model (grapheme-based)
mfa align \
    raw_te/ \
    --phone_set UNICODE \
    aligned_te/
```

---

## 7. English Pipeline

```bash
# Step 1: Preprocess (fix TextGrids, extract features, apply manifest split)
python English/preprocess.py English/config.yaml

# Step 2: Split combined config into three separate YAML files
python make_configs.py English/config.yaml

# Step 3: Train (10,000 steps)
python English/train.py --restore_step 0 \
    -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml

# Step 4: Run inference and generate variance plots
python English/inference.py --restore_step 10000 \
    -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml \
    --out Results/English
```

---

## 8. Telugu Pipeline

```bash
# Step 1: Preprocess (fix TextGrids, build Telugu symbol list, extract features)
#         This regenerates text/symbols.py for Telugu graphemes.
python Telugu/preprocess.py Telugu/config.yaml

# Step 2: Split combined Telugu config
python make_configs.py Telugu/config.yaml

# Step 3: Train (10,000 steps, single-speaker)
python Telugu/train.py --restore_step 0 \
    -p config/preprocess_telugu.yaml -m config/model_telugu.yaml -t config/train.yaml

# Step 4: Run inference and generate variance plots
python Telugu/inference.py --restore_step 10000 \
    -p config/preprocess_telugu.yaml -m config/model_telugu.yaml -t config/train.yaml \
    --out Results/Telugu
```

> **Note:** `Telugu/preprocess.py` overwrites `text/symbols.py` to use Telugu graphemes.  
> To run English and Telugu in the same environment, run Telugu preprocessing in a separate  
> Python session, or restore `text/symbols.py` to the ARPAbet version after Telugu preprocessing.

---

## 9. Training Status

| Experiment | Status |
|---|---|
| English preprocessing | ✅ Completed |
| English training | ✅ 10,000-step checkpoint |
| English audio generation | ✅ 5 WAV samples (E1–E4) |
| English variance plots | ✅ 15 plots (5 sentences × 3 features) |
| Telugu preprocessing | ✅ Completed |
| Telugu training | ⚠️ 1,000-step checkpoint verified; 10,000-step target |
| Telugu inference | ⚠️ Pending 10,000-step checkpoint |

---

## 10. Test Sentences

### English (E1–E4)

| ID | Sentence | Linguistic Feature |
|----|----------|--------------------|
| E1 | "The antidisestablishmentarianism movement..." | Long polysyllabic word |
| E2a | "The orchestra's **conduct** during the ceremony..." | Heteronym: conduct (noun) |
| E2b | "The researchers will **conduct** an experiment..." | Heteronym: conduct (verb) |
| E3 | "The rhyth m of the orchestra..." | Intentional typo/space |
| E4 | "The meeting will start at 10:45 AM on 12 November 2026..." | Numbers, time, date |

### Telugu (T1–T4)

| ID | Sentence | Linguistic Feature |
|----|----------|--------------------|
| T1 | "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో..." | Long compound word |
| T2a | "తాడుపై అతని **పట్టు** చాలా బలంగా ఉంది." | పట్టు = grip (noun) |
| T2b | "ఆమె **పట్టు**చీరను ధరించి వేడుకకు వచ్చింది." | పట్టు = silk (in compound) |
| T3 | "పులి అడవిలో నిశ్శబ్దంగా నడిచింది." | Simple sentence |
| T4 | "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు..." | Numbers, time, ordinal |

---

## 11. Codebase Changes for Telugu Support

The following changes were made to the original ming024/FastSpeech2 codebase to support Telugu:

| Change | Location | Description |
|--------|----------|-------------|
| Telugu symbol list | `text/telugu_symbols.json` | 63 Unicode Telugu graphemes |
| Dynamic symbol loading | `text/symbols.py` | Loads Telugu symbols from JSON |
| Telugu text normaliser | `tts_helpers.py` | Digits/time/ordinals → Telugu words |
| Telugu grapheme converter | `tts_helpers.py` | Maps Unicode chars to symbol sequences |
| Telugu preprocessing | `Telugu/preprocess.py` | Builds symbol set from MFA output |
| Telugu config | `Telugu/config.yaml` | `multi_speaker: false`, `language: te` |
| Telugu inference | `Telugu/inference.py` | 5 test sentences with grapheme front-end |
| Synthesize Telugu | `synthesize.py` | Added `preprocess_telugu()` function |
| Librosa API patch | `audio/stft.py` | `pad_center(size=...)`, `librosa_mel_fn(sr=...)` |
| PyTorch API patch | `utils/model.py` | `torch.load(..., weights_only=False)` |
| Updated dependencies | `requirements.txt` | Actual working library versions |

---

## 12. References

1. Ren, Y. et al. (2021). FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. *ICLR 2021*. https://arxiv.org/abs/2006.04558
2. ming024. FastSpeech2. https://github.com/ming024/FastSpeech2
3. Kong, J. et al. (2020). HiFi-GAN. *NeurIPS 2020*. https://arxiv.org/abs/2010.05646
4. AI4Bharat. Svarah Dataset. https://huggingface.co/datasets/ai4bharat/Svarah
5. SPRINGLab. IndicTTS Telugu. https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
