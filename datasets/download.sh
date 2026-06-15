#!/usr/bin/env bash
# Download the OPTIONAL supporting datasets. The PRIMARY probe is generated
# locally (no download): python datasets/sleep_persona_probe/generate_probe.py
set -e
cd "$(dirname "$0")"

echo "[1/2] PRISM Alignment (real user demographics + conversations)"
python - << 'PY'
from datasets import load_dataset
ds = load_dataset("HannahRoseKirk/prism-alignment", "conversations")
ds.save_to_disk("prism_alignment")
print("saved prism_alignment")
PY

echo "[2/2] OpinionQA persona data -> see code/opinions_qa (CodaLab):"
echo "  https://worksheets.codalab.org/worksheets/0x6fb693719477478aac73fc07db333f69"
