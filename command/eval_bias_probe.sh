#!/bin/bash
# Mounting-bias probe (mechanism control): inject a constant per-take random
# sensor-to-bone foot rotation (per-axis std sigma), three draws per sigma.
set -euo pipefail
cd "$(dirname "$0")/.."

for SIG in 5 10 20 40; do
  for SEED in 20260730 1 2; do
    SUF=""; [ "$SEED" = 1 ] && SUF="s1"; [ "$SEED" = 2 ] && SUF="s2"
    python eval.py --mode finetune --ckpt imuposer_headfeet_all3 --sensors headfeet \
        --foot-bias-noise $SIG --foot-bias-seed $SEED --seed 0 \
        --tag footbias${SIG}${SUF}_run
  done
done

echo "Per-sigma summary (mean +- sd over draws):"
python - <<'EOF'
import json
import numpy as np
from oswb import config

for sig in (5, 10, 20, 40):
    vals = []
    for suf in ("", "s1", "s2"):
        p = config.RESULTS_DIR / "eval" / f"cv_footbias{sig}{suf}_run.json"
        vals.append(json.load(open(p))["summary"]["ALL"]["rigid_mpjpe"])
    v = np.array(vals)
    print(f"sigma={sig:2d}deg | {' '.join(f'{x:6.1f}' for x in v)} | {v.mean():5.1f} +- {v.std(ddof=1):.1f}")
print(" sigma= 0deg |  95.8 (no injected bias; cv_finetune_all3)")
EOF
