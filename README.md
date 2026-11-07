# FastSpeech 2 for English and Telugu Text-to-Speech

**Author:** Abhiram K

## 1. Project Overview

This project explores neural Text-to-Speech (TTS) synthesis using FastSpeech 2 for English and Telugu.
The objective is to convert written text into speech using language-specific acoustic models.

The experiments use the Svarah English dataset and IndicTTS Telugu dataset.

## 2. Model Architecture

FastSpeech 2 is a non-autoregressive neural TTS architecture.
Its variance adaptor predicts:
- Duration
- Pitch
- Energy

The model generates mel-spectrograms, which are converted into speech waveforms using HiFi-GAN.

## 3. Project Workflow

1. Dataset preparation
2. Text cleaning and preprocessing
3. Acoustic feature extraction
4. FastSpeech 2 model training
5. Mel-spectrogram prediction
6. HiFi-GAN waveform generation
7. Audio and variance-adaptor result analysis

## 4. Technologies Used

- Python
- PyTorch
- FastSpeech 2
- HiFi-GAN
- Librosa
- NumPy
- Matplotlib
- Google Colab

## 5. Repository Structure

```text
English/          English experiment files
Telugu/           Telugu experiment files
audio/            Audio processing and feature extraction
config/           Model, preprocessing and training configurations
hifigan/          HiFi-GAN vocoder implementation
model/            FastSpeech 2 model components
preprocessor/     Dataset preprocessing utilities
text/             Text processing and phoneme utilities
transformer/      Transformer architecture components
utils/            Shared utility functions
Results/          Generated outputs and visualizations
Report/           Technical report
README.md         Project documentation
requirements.txt  Python dependencies
.gitignore        Excluded files
```

## 6. Dataset Information

### English

The English experiment uses the Svarah dataset. Preprocessing includes text cleaning and acoustic feature extraction.

### Telugu

The Telugu experiment uses the IndicTTS dataset. Preprocessing includes text and alignment processing and extraction of mel, pitch, energy and duration features.

The original datasets are not included in this repository and must be obtained separately.

## 7. Training Status

| Experiment | Verified status |
|---|---|
| English preprocessing | Completed |
| English training | 10,000-step checkpoint verified |
| English audio generation | Five WAV samples generated |
| English variance plots | 15 plots generated |
| Telugu preprocessing | Completed |
| Telugu training | 1,000-step checkpoint verified |
| Telugu 10,000-step training | Not yet verified |
| Telugu inference | Not yet verified |

## 8. Results and Observations

Five English WAV samples were generated and basic waveform validity was checked.
However, listening evaluation indicated that the generated speech was not sufficiently intelligible and requires improvement.

Duration, pitch and energy visualizations were generated to inspect the variance-adaptor predictions.

## 9. Installation

Clone the repository:

```bash
git clone https://github.com/abhiram-k-dev/fastspeech2-english-telugu-tts.git
cd fastspeech2-english-telugu-tts
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Obtain the required datasets, configure their paths and prepare the acoustic features before running training or inference.
Training and synthesis also require the appropriate model configuration and vocoder checkpoint.

## 10. Limitations and Future Work

- Improve English speech intelligibility.
- Complete and verify Telugu training up to the target of 10,000 steps.
- Verify Telugu inference and audio generation.
- Evaluate speech quality using objective metrics and human listening tests.
- Perform additional tests with different input sentences.

## 11. Technical Report

Refer to the Report directory for the technical report describing the experiments, implementation, results and limitations.

## 12. Acknowledgements

This project is based on the FastSpeech 2 architecture and uses HiFi-GAN for waveform generation.

---

**Status:** Experimental implementation. The training and evaluation status above reflects the results verified during development.
