"""Usage: python make_configs.py config.yaml -> writes the three required YAML files."""
import sys
import tts_helpers as h

print(
    "wrote config/ from",
    sys.argv[1] if len(sys.argv) > 1 else "config.yaml",
    h.split_config(*(sys.argv[1:2] or ["config.yaml"]))
)
