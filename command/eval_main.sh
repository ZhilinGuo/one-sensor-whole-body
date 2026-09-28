#!/bin/bash
# Main results (Table 1): controls + IMUPoser-adapted + MobilePoser-adapted,
# leave-one-run-out (run) and leave-one-motion-out (seq) splits.
# Requires pretrained checkpoints (command/pretrain.sh) and data/processed.
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- controls ----
# Constant mean-pose prior (no model)
python eval.py --mode prior --tag prior
# Zero-input control (all sensor channels masked)
python eval.py --mode finetune --ckpt imuposer_headfeet_all3 --sensors none \
    --tag finetune_noinput_all3
# No-calibration control (raw device-frame features)
python eval.py --mode finetune --ckpt imuposer_headfeet_all3 --calib-mode raw \
    --tag finetune_nocalib_all3

# ---- IMUPoser-adapted ----
for SPLIT in run seq; do
  PRE=""; [ "$SPLIT" = seq ] && PRE="seqout_"
  python eval.py --mode finetune --ckpt imuposer_headfeet_all3 --sensors head \
      --split $SPLIT --tag finetune_${PRE}head_all3
  python eval.py --mode finetune --ckpt imuposer_headfeet_all3 --sensors feet \
      --split $SPLIT --tag finetune_${PRE}feet_all3
  python eval.py --mode finetune --ckpt imuposer_headfeet_all3 --sensors headfeet \
      --split $SPLIT --tag finetune_${PRE}all3
  # causal (streaming) head-only variant
  python eval.py --mode finetune --ckpt imuposer_causal_all3 --sensors head \
      --split $SPLIT --causal --tag causal_${PRE}head_all3
done

# ---- MobilePoser-adapted ----
for SPLIT in run seq; do
  PRE=""; [ "$SPLIT" = seq ] && PRE="seqout_"
  python mobileposer.py cv --ckpt mobileposer_headfeet_all3 --sensors head \
      --split $SPLIT --tag mp_${PRE}head_all3
  python mobileposer.py cv --ckpt mobileposer_headfeet_all3 --sensors headfeet \
      --split $SPLIT --tag mp_${PRE}all3
done
