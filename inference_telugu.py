"""Telugu inference. Usage (from the code folder):
python inference_telugu.py --restore_step 10000 -p config/preprocess.yaml -m config/model.yaml -t config/train.yaml --out Results/Telugu
VERIFY the Telugu strings below against the original assignment PDF (the PDF text extraction garbled them)."""
import argparse, json, os, sys
import torch, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tts_helpers as h
from utils.model import get_model, get_vocoder

TEST = {  # reconstruction of the intended text (UNVERIFIED, check against the PDF)
    "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    "T2b": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.",
    "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
    "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
}
EXTRA_AS_EXTRACTED = {  # strings exactly as the PDF text layer gave them (garbled)
    "T1_extracted": "న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలోనూతన ఔషధాల ప్రభావాన్ని విశ్లషే ించారు.",
    "T3_extracted": "పులి అడ విలో నిశ్శబ్ంద గా నడిచింది.",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore_step", type=int, required=True)
    ap.add_argument("-p", "--preprocess_config", required=True)
    ap.add_argument("-m", "--model_config", required=True)
    ap.add_argument("-t", "--train_config", required=True)
    ap.add_argument("--out", default="Results/Telugu")
    ap.add_argument("--also_extracted", action="store_true")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pc, mc, tc = [yaml.load(open(f, encoding="utf-8"), Loader=yaml.FullLoader) for f in (a.preprocess_config, a.model_config, a.train_config)]
    configs = (pc, mc, tc)
    model = get_model(a, configs, device, train=False)
    vocoder = get_vocoder(mc, device)
    symbols = set(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "text", "telugu_symbols.json"), encoding="utf-8")))
    font = h.find_telugu_font()
    print("Telugu font for plots:", font)
    items = dict(TEST)
    if a.also_extracted:
        items.update(EXTRA_AS_EXTRACTED)
    for name, text in items.items():
        ph = h.te_phonemes(text, symbols)
        print(name, "|", text, "\n   normalised:", h.normalize_te(text), "\n   ", ph)
        h.synthesize_and_plot(ph, name, text, 0, model, vocoder, configs, device,
                              f"{a.out}/test_generated_audios", f"{a.out}/variance_adaptor_plots", font=font)
        open(f"{a.out}/test_generated_audios/{name}_phonemes.txt", "w", encoding="utf-8").write(ph)
