"""Leave-one-run-out / leave-one-motion-out evaluation of the sparse-IMU model.

Three modes:
  * ``zeroshot`` : evaluate the synthetic-pretrained model directly on real takes.
  * ``finetune`` : for each held-out group, fine-tune the pretrained model on the
                   other groups (masked, frame-aligned joint loss), then evaluate.
  * ``prior``    : constant mean-pose baseline (no model; Table 1 first row).

Predictions are SMPL joint rotations -> FK -> joint positions, scored against the
pseudo-GT joints. Because real IMU calibration leaves a global frame/scale
ambiguity, the primary metric is rigid-MPJPE (a single rotation+scale per
sequence); we also report PA-MPJPE (per-frame Procrustes), motion R^2 against the
temporal mean pose, MPJVE and jitter.

    python eval.py --mode finetune --ckpt imuposer_headfeet_all3 --sensors head
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from oswb import config
from oswb import jointset
from oswb.model import PoseRNN, N_INPUT
from oswb.cv import (
    evaluate, evaluate_prior, finetune, global_mean_pose, summarize,
    _inject_foot_bias,
)
from oswb.dataset import load_all_labeled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["zeroshot", "finetune", "prior"], default="finetune")
    ap.add_argument("--ckpt", default="imuposer_headfeet_all3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--win", type=int, default=150)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--calib-mode", default="full", choices=["full", "raw"],
                    help="'raw' = no-calibration control (raw device-frame features)")
    ap.add_argument("--sensors", default="headfeet",
                    choices=["headfeet", "head", "feet", "lfoot", "rfoot", "none"])
    ap.add_argument("--split", default="run", choices=["run", "seq"],
                    help="leave-one-{run,seq}-out")
    ap.add_argument("--joints", default="lower", choices=["lower", "full", "fulls"],
                    help="model/eval joint set (full = 20-joint full-body extension; "
                         "fulls = full + weak spine supervision, spines not evaluated)")
    ap.add_argument("--causal", action="store_true",
                    help="unidirectional LSTM (streaming variant); needs a causal pretrained ckpt")
    ap.add_argument("--foot-bias-noise", type=float, default=0.0, metavar="DEG",
                    help="per-take constant foot-orientation mounting bias ~ N(0, DEG^2) per axis; "
                         "simulates session-to-session sensor-to-bone calibration spread")
    ap.add_argument("--foot-bias-seed", type=int, default=20260730,
                    help="RNG seed for --foot-bias-noise draws (vary for multi-draw sweeps)")
    ap.add_argument("--stage-lower-epochs", type=int, default=0, metavar="K",
                    help="staged fine-tuning: first K epochs lower-body-only loss "
                         "(upper weights zeroed), then full weighted loss")
    ap.add_argument("--freeze-encoder-stage2", action="store_true",
                    help="freeze encoder (linear1+rnn) during stage 2; only the "
                         "output head adapts to the widened task")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch/numpy RNGs for reproducible fine-tuning "
                         "(default: unseeded, single random instance)")
    a = ap.parse_args()
    if a.seed is not None:
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)

    js = jointset.get(a.joints)
    dev = a.device if torch.cuda.is_available() else "cpu"

    def fresh_model():
        ck = None
        if a.ckpt.lower() != "none":
            ck = torch.load(config.RESULTS_DIR / "baselines" / a.ckpt / "ckpt.pt",
                            map_location=dev, weights_only=False)
        n_in = ck.get("n_input", N_INPUT) if ck is not None else N_INPUT
        m = PoseRNN(n_in, js.n_output, bidirectional=not a.causal).to(dev)
        if ck is not None:
            m.load_state_dict(ck["model"])
        return m

    takes = load_all_labeled(calib_mode=a.calib_mode, sensors=a.sensors, joints=a.joints)
    if a.foot_bias_noise > 0:
        _inject_foot_bias(takes, a.foot_bias_noise, seed=a.foot_bias_seed)
    # Restrict targets to the supervised joints once; FK outputs are sliced with
    # the same indices, so loss/eval never see the unsupervised spine slots.
    sup = np.asarray(js.sup_idx)
    for ta in takes:
        ta.Yj = ta.Yj[:, sup]
    # Metrics use only the evaluated subset (loss-only spine slots excluded).
    eval_set = set(js.eval_idx)
    eval_pos = [p for p, out_i in enumerate(js.sup_idx) if out_i in eval_set]
    if eval_pos == list(range(len(js.sup_idx))):
        eval_pos = None  # everything supervised is evaluated (lower / full sets)
    group_pos = None
    if js.name.startswith("full"):
        pos = {out_i: p for p, out_i in enumerate(js.eval_idx)}
        group_pos = {g: [pos[i] for i in idx if i in pos] for g, idx in js.groups.items()}
    mean_pose = global_mean_pose(takes)
    if eval_pos is not None:
        mean_pose = mean_pose[eval_pos]
    eval_names = [js.names[i] for i in js.eval_idx]
    key = (lambda t: t.run) if a.split == "run" else (lambda t: t.seq)
    groups = sorted({key(t) for t in takes})
    print(f"[cv] {len(takes)} takes; split=leave-one-{a.split}-out groups={groups}; mode={a.mode}; "
          f"dev={dev}; calib={a.calib_mode}; sensors={a.sensors}; ckpt={a.ckpt}", flush=True)

    rows = []
    if a.mode == "prior":
        rows = evaluate_prior(takes, mean_pose, js, group_pos, eval_pos)
    elif a.mode == "zeroshot":
        rows = evaluate(fresh_model(), takes, dev, mean_pose, js, group_pos, eval_pos)
    else:
        for g in groups:
            tr = [t for t in takes if key(t) != g]
            te = [t for t in takes if key(t) == g]
            if not te:
                continue
            model = finetune(fresh_model(), tr, dev, a.epochs, a.lr, a.batch, a.win, js,
                             stage_lower=a.stage_lower_epochs,
                             freeze_stage2=a.freeze_encoder_stage2)
            fold = evaluate(model, te, dev, mean_pose, js, group_pos, eval_pos)
            print(f"  [fold {a.split}{g}] PA={np.mean([x['pa_mpjpe'] for x in fold]):.1f}mm "
                  f"rigid={np.mean([x['rigid_mpjpe'] for x in fold]):.1f}mm (n={len(fold)})", flush=True)
            rows += fold

    tag = a.tag or a.mode
    summ = summarize(rows, tag, eval_names, groups=group_pos)
    out_dir = config.RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"cv_{tag}.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=2))
    print(f"\n[cv] wrote {out_dir / f'cv_{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
