#!/bin/bash
# Paired per-take significance tests (head-only vs head+feet, 2 model families x
# 2 splits, Holm-corrected) + the 2x2 paired-scatter figure.
# Requires the eight cv_*.json evaluations from command/eval_main.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

python significance.py
