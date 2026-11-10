# FastSpeech 2 Text-to-Speech: English and Telugu

**Author:** Abhiram K  
**Assignment:** FastSpeech 2 TTS Implementation and Evaluation  
**Reference Paper:** [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558)  
**Reference Implementation:** [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)

---

## 1. Introduction

This report documents the implementation and evaluation of FastSpeech 2 Text-to-Speech (TTS) models for two languages: English (Indian-accented) and Telugu. The goal is to understand the FastSpeech 2 architecture, adapt the existing codebase independently for each language, train two separate models on the specified datasets, generate speech samples, and analyse the variance adaptor predictions (duration, pitch, energy).

---

## 2. Model Architecture

FastSpeech 2 is a **non-autoregressive** (parallel) neural TTS model. Unlike autoregressive models (Tacotron, WaveNet) that generate audio one frame at a time, FastSpeech 2 predicts the entire mel-spectrogram in a single forward pass, making inference significantly faster.

### 2.1 Overall Pipeline

```
Input text
  ↓  [Text normalisation]
  ↓  [Phoneme/grapheme conversion]
  ↓  [Symbol-to-ID embedding + positional encoding]
  
[Encoder]  — 4 × Feed-Forward Transformer (FFT) blocks
  ↓  hidden: 256-dim, 2 attention heads, FFT conv kernel [9,1]

[Variance Adaptor]
  ├─ Duration Predictor → length regulator (expand encoder output frame-by-frame)
  ├─ Pitch Predictor → pitch embedding (phoneme-level F0)
  └─ Energy Predictor → energy embedding (phoneme-level RMS energy)
  
[Decoder]  — 6 × Feed-Forward Transformer (FFT) blocks
  ↓  hidden: 256-dim, 2 attention heads

[Linear layer] → 80-band mel-spectrogram

[PostNet]  — 5-layer 1-D convolution refinement network

[HiFi-GAN universal vocoder] → 22050 Hz waveform
```

### 2.2 Feed-Forward Transformer (FFT) Block

Each FFT block consists of:
1. **Multi-head self-attention** (2 heads, 256-dim key/value/query)
2. **1-D convolution** (filter size 1024, kernel sizes [9, 1])
3. Layer normalisation + residual connections
4. Dropout (0.2 for encoder/decoder, 0.5 for variance predictor)

### 2.3 Variance Adaptor

The variance adaptor is FastSpeech 2's key contribution over FastSpeech 1. It predicts three prosodic features that control the naturalness of speech:

| Predictor | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Duration** | Encoder hidden states | Log duration per phoneme (frames) | Controls speaking rate; replaces the need for an external aligner at inference |
| **Pitch** | Encoder hidden states + duration-expanded | F0 per phoneme (Hz, normalised) | Controls fundamental frequency / intonation |
| **Energy** | Encoder hidden states + duration-expanded | RMS energy per phoneme (normalised) | Controls loudness and emphasis |

Both pitch and energy are computed at **phoneme level** (averaged over frames) rather than frame level, which produces more stable predictions.

---

## 3. Datasets

### 3.1 English — Svarah Dataset

- **Source:** [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Language:** Indian-accented English
- **Content:** Multi-speaker read speech, covering various English accents from Indian speakers
- **Used subset:** ~5 hours of clean speech
- **Speakers:** Multiple (115 pseudo-speakers in the full dataset)
- **Sampling rate:** 22050 Hz (resampled from original)
- **Transcription format:** Text files (.lab) with one sentence per file

### 3.2 Telugu — IndicTTS Dataset

- **Source:** [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)
- **Language:** Telugu (Dravidian language, ~82 million speakers)
- **Content:** Single-speaker studio-quality read speech
- **Used subset:** ~5 hours
- **Speakers:** Single speaker (female)
- **Sampling rate:** 22050 Hz
- **Transcription format:** Text files (.lab) in UTF-8 Telugu Unicode

---

## 4. Data Preparation

### 4.1 Forced Alignment with MFA

Both datasets require **forced alignment** to obtain phone-level duration boundaries (TextGrids). The [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/) was used.

**English alignment:**
- MFA acoustic model: `english_us_arpa`
- Dictionary: Combined CMU pronunciation dictionary + Svarah-specific entries (`en_merged.dict`)
- Output: ARPAbet phone labels with timing (TextGrids)

**Telugu alignment:**
- MFA approach: Grapheme-level alignment (each Telugu Unicode character treated as a "phone")
- No pre-built Telugu phoneme dictionary → used grapheme-to-grapheme alignment
- Output: Telugu Unicode character labels with timing (TextGrids)

### 4.2 TextGrid Preprocessing

A common issue with MFA output is empty silence labels (blank strings instead of `sp`/`sil`). The `fix_textgrids()` function in `tts_helpers.py` renames all empty labels to `'sp'` before feature extraction.

```python
# tts_helpers.py — fix_textgrids()
for iv in tier._objects:
    if iv.text.strip() == "":
        iv.text = "sp"
```

### 4.3 Feature Extraction

For each utterance, the following features are extracted:
- **Mel-spectrogram:** 80 bands, FFT 1024, hop 256, window 1024, 22050 Hz
- **Pitch (F0):** PyWorld HARVEST + StoneMask, interpolated over voiced regions, averaged to phoneme level
- **Energy:** RMS energy of mel-spectrogram, averaged to phoneme level
- **Duration:** Number of mel frames per phoneme from TextGrid timing

All features are normalised to zero mean, unit variance across the training set.

### 4.4 Train/Validation Split

- Both models use the split defined in the manifest CSV (`split` column: `train`/`val`).
- The random split from the original `Preprocessor.build_from_path()` is overwritten by `rewrite_split()`.

---

## 5. Text Front-End

### 5.1 English Text Front-End

**Pipeline:** Raw text → Normalisation → G2P → ARPAbet phoneme sequence

**Text normalisation** (`normalize_en` in `tts_helpers.py`):
- Times: `10:45 AM` → `ten forty-five a m`
- Years: `2026` → `twenty twenty-six`
- Numbers: `12` → `twelve`
- Symbols: `&` → `and`, `%` → `percent`

**Grapheme-to-Phoneme (G2P):**
- Primary: Pronunciation dictionary lookup (`en_merged.dict`)
- Fallback: `g2p_en` neural G2P model for out-of-vocabulary words
- Heteronym handling: Manual overrides per word per sentence (e.g., "conduct" noun vs. verb)

**Symbol set:** ARPAbet phones + stress marks (e.g., `AH0`, `AH1`, `AH2`), special tokens `sp`, `sil`, `spn`

### 5.2 Telugu Text Front-End

**Pipeline:** Raw text → Normalisation → Grapheme sequence (direct)

**Text normalisation** (`normalize_te` in `tts_helpers.py`):
- Times: `10:45` → `పది నలభై ఐదు` (ten forty-five in Telugu)
- Ordinals: `12వ` → `పన్నెండవ`
- Numbers: `2026` → `రెండు వేల ఇరవై ఆరు`
- Unicode normalisation: NFC + zero-width character removal

**Grapheme processing:**
- Each Unicode Telugu character maps directly to a symbol (no phoneme conversion)
- 63 Telugu Unicode grapheme symbols (vowels, consonants, matras, anusvara, visarga)
- Spaces and punctuation → `sp` (silence token)

**Key difference from English:** Telugu uses a **grapheme-to-grapheme** approach. This works well for Telugu because Telugu has a highly regular orthography where the grapheme-to-phoneme mapping is mostly one-to-one (unlike English). This avoids the need for a Telugu phoneme dictionary.

---

## 6. Codebase Modifications for Telugu

The following changes were made to the original [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) codebase to support Telugu:

| File | Change | Reason |
|------|--------|--------|
| `text/symbols.py` | Completely rewritten to load Telugu Unicode graphemes from JSON | ARPAbet symbols not applicable for Telugu |
| `text/telugu_symbols.json` | New file — 63 Telugu Unicode characters | Defines the Telugu symbol vocabulary |
| `tts_helpers.py` | New file — Telugu normaliser, grapheme converter, text front-end | Language-specific text processing |
| `Telugu/preprocess.py` | New file — Telugu preprocessing with grapheme symbol building | Telugu requires dynamic symbol set from MFA output |
| `Telugu/config.yaml` | New file — Telugu-specific configuration | `multi_speaker: false`, `language: te`, no lexicon path |
| `Telugu/train.py` | New file — Telugu training entry point | Documents Telugu training procedure |
| `Telugu/inference.py` | New file — Telugu inference with 5 test sentences | Documents Telugu inference procedure |
| `config/preprocess_telugu.yaml` | New file — separate Telugu preprocess config | Avoids overwriting English config |
| `config/model_telugu.yaml` | New file — Telugu model config | `multi_speaker: false` |
| `synthesize.py` | Added `preprocess_telugu()` function + `elif lang == 'te'` branch | Telugu single-sentence synthesis |
| `requirements.txt` | Updated to reflect actual library versions used | Old versions (torch 1.7, librosa 0.7.2) incompatible with Colab |
| `audio/stft.py` | Patched `pad_center(size=...)` and `librosa_mel_fn(sr=..., n_fft=...)` | API changes in librosa ≥ 0.9 |
| `utils/model.py` | Patched `torch.load(..., weights_only=False)` | Required by PyTorch ≥ 2.0 |

---

## 7. Training

### 7.1 Hyperparameters (Both Languages)

| Parameter | Value |
|-----------|-------|
| Encoder layers | 4 |
| Decoder layers | 6 |
| Attention heads | 2 |
| Hidden dimension | 256 |
| Conv filter size | 1024 |
| Conv kernel sizes | [9, 1] |
| Encoder dropout | 0.2 |
| Decoder dropout | 0.2 |
| Variance predictor dropout | 0.5 |
| Pitch/energy quantisation | Linear (256 bins) |
| Batch size | 16 |
| Warm-up steps | 1000 |
| Anneal steps | [6000, 8000, 9000] |
| Anneal rate | 0.3 |
| Total steps | 10000 |
| Checkpoint interval | 1000 steps |

### 7.2 English Training Results

| Metric | Value |
|--------|-------|
| Final checkpoint | 10,000 steps |
| Training completed | Yes |
| Speech quality | Partially intelligible (Indian-accented English with some artefacts) |
| WAV samples generated | 5 (E1–E4) |
| Variance plots generated | 15 (5 sentences × 3 features) |

### 7.3 Telugu Training Results

| Metric | Value |
|--------|-------|
| Verified checkpoint | 1,000 steps |
| Target checkpoint | 10,000 steps |
| Training completed | Not yet verified |
| Inference verified | Not yet verified |

> **Note:** Telugu training requires a full GPU session. The 10,000-step checkpoint should be generated by running `Telugu/train.py --restore_step 1000` on a Colab GPU runtime with the IndicTTS dataset.

---

## 8. Variance Adaptor Analysis

The variance adaptor produces three outputs at inference time, which reveal how the model interprets the input text. For each test sentence, duration, pitch, and energy plots were generated and saved to `Results/<Language>/variance_adaptor_plots/`.

### 8.1 English Observations

**E1 — Long polysyllabic word ("antidisestablishmentarianism"):**
- Duration: longer durations on stressed syllables, shorter on unstressed
- Pitch: peaks on lexically stressed syllables (steeper contour)
- Energy: higher at word onset, gradual decay

**E2a vs E2b — Heteronym "conduct":**
- Different phoneme sequences were fed (noun: `K AA1 N D AH0 K T`, verb: `K AH0 N D AH1 K T`)
- Expected: different pitch/energy on the stressed syllable
- Duration: verb "conduct" shows longer duration on the second syllable

**E3 — Intentional typo ("rhyth m" with space):**
- The space caused the two fragments ("rhyth" and "m") to be treated as separate words
- G2P produces different phoneme sequences for each fragment vs. the whole word
- Visible in duration plot as an unusual pause boundary

**E4 — Mixed content (time, number, date):**
- Text normalisation expanded "10:45 AM" → "ten forty-five a m" before G2P
- Longer duration on content words (November), shorter on function words

### 8.2 Telugu Observations

*(To be filled in after completing Telugu training and inference.)*

For each of the 5 Telugu test sentences, the following are expected:
- T1 (long compound): high energy on compound constituents; multiple duration peaks
- T2a/T2b (polysemy "పట్టు"): same grapheme, potentially similar predictions since the model does not have semantic disambiguation
- T3 (simple sentence): smoother pitch contour
- T4 (mixed with numerals): text normalisation converts `2026`, `12వ`, `10:45` to Telugu words before grapheme extraction

---

## 9. How to Reproduce

### Prerequisites

```bash
# 1. Clone the repository
git clone https://github.com/abhiram-k-dev/fastspeech2-english-telugu-tts.git
cd fastspeech2-english-telugu-tts

# 2. Install dependencies (Python 3.8–3.10, GPU recommended)
pip install -r requirements.txt

# 3. Download HiFi-GAN universal vocoder
#    Place generator_universal.pth.tar and config.json in hifigan/
#    Available from https://github.com/jik876/hifi-gan/releases

# 4. Download datasets
#    Svarah: https://huggingface.co/datasets/ai4bharat/Svarah
#    IndicTTS Telugu: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
```

### English Pipeline

```bash
# Step 1: Run MFA forced alignment (run separately)
#   mfa align raw_en/ en_merged.dict english_us_arpa aligned_en/

# Step 2: Preprocess
python English/preprocess.py English/config.yaml

# Step 3: Split config (English/config.yaml → config/*.yaml)
python make_configs.py English/config.yaml

# Step 4: Train
python English/train.py --restore_step 0 \
    -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml

# Step 5: Inference (after training)
python English/inference.py --restore_step 10000 \
    -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml \
    --out Results/English
```

### Telugu Pipeline

```bash
# Step 1: Run MFA forced alignment with grapheme model
#   mfa align raw_te/ --phone_set UNICODE aligned_te/

# Step 2: Preprocess (also rebuilds text/symbols.py for Telugu)
python Telugu/preprocess.py Telugu/config.yaml

# Step 3: Split config
python make_configs.py Telugu/config.yaml config/

# Step 4: Train
python Telugu/train.py --restore_step 0 \
    -p config/preprocess.yaml -m config/model_telugu.yaml -t config/train.yaml

# Step 5: Inference (after training to 10000 steps)
python Telugu/inference.py --restore_step 10000 \
    -p config/preprocess_telugu.yaml -m config/model_telugu.yaml -t config/train.yaml \
    --out Results/Telugu

# Optional: install Telugu font for readable axis labels
#   sudo apt-get install -y fonts-telu   # on Ubuntu/Colab
```

---

## 10. Limitations and Future Work

1. **Telugu intelligibility:** The grapheme-level approach may limit the model's ability to handle complex Telugu conjunct consonants (ottulu). A phoneme-level representation using a Telugu pronunciation dictionary would likely improve quality.

2. **English intelligibility:** The 10,000-step English model produces partially intelligible speech. Longer training (50,000–100,000 steps) or a larger batch size would likely improve quality significantly.

3. **Speaker diversity for Telugu:** IndicTTS Telugu is single-speaker. Training on a multi-speaker corpus would generalise better.

4. **Objective metrics:** Future work should include MOS (Mean Opinion Score) listening tests and objective metrics such as MCD (Mel Cepstral Distortion) to quantitatively evaluate synthesis quality.

5. **Number/ordinal normalisation coverage:** The Telugu normaliser covers common cases but may fail on rare ordinals, large numbers (>9999), or Sanskrit loan words.

---

## 11. References

1. Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T. Y. (2021). FastSpeech 2: Fast and High-Quality End-to-End Text to Speech. *ICLR 2021*. https://arxiv.org/abs/2006.04558

2. ming024. FastSpeech2 (GitHub implementation). https://github.com/ming024/FastSpeech2

3. Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. *NeurIPS 2020*. https://arxiv.org/abs/2010.05646

4. McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). Montreal Forced Aligner. *Interspeech 2017*.

5. AI4Bharat. Svarah Dataset. https://huggingface.co/datasets/ai4bharat/Svarah

6. SPRINGLab. IndicTTS Telugu Dataset. https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
