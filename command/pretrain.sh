#!/bin/bash
# Synthetic pretraining for all model variants (40 epochs each, ~hours on one GPU).
# Requires data/synthetic/AMASS (run synth_amass.py first) and data/smpl.
set -euo pipefail
cd "$(dirname "$0")/.."

# IMUPoser-adapted BiLSTM, head+feet input layout (main model)
python pretrain.py --name imuposer_headfeet_all3

# Causal (unidirectional) streaming variant
python pretrain.py --name imuposer_causal_all3 --causal

# 20-joint full-body extension
python pretrain.py --name imuposer_full20_all3 --joints full

# MobilePoser-adapted two-stage baseline
python mobileposer.py pretrain --name mobileposer_headfeet_all3
