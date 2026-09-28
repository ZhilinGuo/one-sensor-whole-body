#!/bin/bash
# Full-body extension (20-joint output, head-only): naive joint training vs
# weak spine supervision vs staged fine-tuning, three seeds, leave-one-run-out.
set -euo pipefail
cd "$(dirname "$0")/.."

for SEED in 0 1 2; do
  # naive: direct 20-joint training (legs degrade to the static prior)
  python eval.py --mode finetune --ckpt imuposer_full20_all3 --joints full \
      --sensors head --seed $SEED --tag full_naive_s${SEED}
  # weak spine supervision (fulls): constrains the torso FK chain
  python eval.py --mode finetune --ckpt imuposer_full20_all3 --joints fulls \
      --sensors head --seed $SEED --tag fulls_s${SEED}
  # staged: 10 lower-only epochs, then 10 full weighted-loss epochs
  python eval.py --mode finetune --ckpt imuposer_full20_all3 --joints full \
      --sensors head --stage-lower-epochs 10 --seed $SEED --tag full_staged_s${SEED}
done

# staged with frozen trunk (single instance): legs protected, arm signal lost
python eval.py --mode finetune --ckpt imuposer_full20_all3 --joints full \
    --sensors head --stage-lower-epochs 10 --freeze-encoder-stage2 --seed 0 \
    --tag full_staged_frozen_s0
