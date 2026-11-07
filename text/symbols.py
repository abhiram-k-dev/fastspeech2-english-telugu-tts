"""Telugu grapheme symbols (generated from the MFA phones tier)."""
import json, os
_pad = "_"
_te = json.load(open(os.path.join(os.path.dirname(__file__), "telugu_symbols.json"), encoding="utf-8"))
symbols = [_pad] + ["@" + c for c in _te] + ["@sp", "@spn", "@sil"]
