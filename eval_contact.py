"""Cross-validation for auxiliary left/right foot-contact prediction.

Reuses the pose BiLSTM and the pseudo-GT contact labels attached to the
benchmark takes. The pose loss keeps the representation anchored while a small
binary head learns left/right foot-contact probabilities.

    python eval_contact.py --ckpt imuposer_headfeet_all3 --sensors head
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from oswb import config
from oswb import jointset
from oswb.model import ContactPoseRNN, N_INPUT, N_OUTPUT
from oswb import metrics
from oswb import fk
from oswb.dataset import load_all_labeled
from oswb.cv import aligned_joint_loss, pa_mpjpe_fast, rigid_mpjpe

JS = jointset.get("lower")


class ContactWindowDataset(Dataset):
    """Fixed windows with pose and binary contact targets."""

    def __init__(self, takes, win=150, stride=75):
        self.win = win
        self.Xs, self.Ys, self.Cs, self.Ms, self.index = [], [], [], [], []
        for k, ta in enumerate(takes):
            if ta.contacts is None:
                continue
            T = ta.X.shape[0]
            self.Xs.append(torch.from_numpy(ta.X))
            self.Ys.append(torch.from_numpy(ta.Yj))
            self.Cs.append(torch.from_numpy(ta.contacts.astype(np.float32)))
            self.Ms.append(torch.from_numpy(ta.valid.astype(np.float32)))
            for st in range(0, max(1, T - win + 1), stride):
                self.index.append((len(self.Xs) - 1, st))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        k, st = self.index[idx]
        x = self.Xs[k][st:st + self.win]
        y = self.Ys[k][st:st + self.win]
        c = self.Cs[k][st:st + self.win]
        m = self.Ms[k][st:st + self.win]
        if x.shape[0] < self.win:
            pad = self.win - x.shape[0]
            x = torch.cat([x, torch.zeros(pad, x.shape[1])])
            y = torch.cat([y, torch.zeros(pad, *y.shape[1:])])
            c = torch.cat([c, torch.zeros(pad, c.shape[1])])
            m = torch.cat([m, torch.zeros(pad)])
        return x, y, c, m


def fresh_model(ckpt: str, device: str) -> ContactPoseRNN:
    model = ContactPoseRNN(N_INPUT, N_OUTPUT).to(device)
    if ckpt.lower() != "none":
        state = torch.load(config.RESULTS_DIR / "baselines" / ckpt / "ckpt.pt",
                           map_location=device, weights_only=False)["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[contact] loaded pose ckpt; missing={list(missing)} unexpected={list(unexpected)}", flush=True)
    return model


def finetune_contact(model, train_takes, device, epochs, lr, batch, win, contact_weight):
    ds = ContactWindowDataset(train_takes, win=win)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=len(ds) > batch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for _ in range(epochs):
        for X, Yj, C, M in dl:
            X, Yj, C, M = X.to(device), Yj.to(device), C.to(device), M.to(device)
            opt.zero_grad()
            pose, logits = model(X)
            loss_pose = aligned_joint_loss(pose, Yj, M, device, JS)
            cmask = M[..., None].expand_as(C)
            loss_contact = (bce(logits, C) * cmask).sum() / (cmask.sum() + 1e-6)
            loss = loss_pose + contact_weight * loss_contact
            loss.backward()
            opt.step()
    return model


def evaluate(model, takes, device, threshold):
    model.eval()
    rows = []
    pred_by = collections.defaultdict(list)
    gt_by = collections.defaultdict(list)
    for ta in takes:
        if ta.contacts is None:
            continue
        X = torch.from_numpy(ta.X)[None].to(device)
        with torch.no_grad():
            pose, logits = model(X)
        J = fk.legs6d_to_joints(pose[0].cpu().reshape(-1, 9, 6)).numpy()
        probs = torch.sigmoid(logits[0]).cpu().numpy()
        pred = probs >= threshold
        v = ta.valid
        pred_by["ALL"].append(pred[v]); gt_by["ALL"].append(ta.contacts[v])
        pred_by[ta.motion_class].append(pred[v]); gt_by[ta.motion_class].append(ta.contacts[v])
        prf = metrics.contact_prf1(pred[v], ta.contacts[v])
        rows.append({
            "run": ta.run,
            "seq": ta.seq,
            "motion": ta.motion_class,
            "rigid_mpjpe": rigid_mpjpe(J[v], ta.Yj[v]),
            "pa_mpjpe": pa_mpjpe_fast(J[v], ta.Yj[v]),
            "contact_precision": prf["precision"],
            "contact_recall": prf["recall"],
            "contact_f1": prf["f1"],
            "contact_accuracy": prf["accuracy"],
            "n": int(v.sum()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="imuposer_headfeet_all3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--win", type=int, default=150)
    ap.add_argument("--contact-weight", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--sensors", default="head",
                    choices=["headfeet", "head", "feet", "lfoot", "rfoot", "none"])
    ap.add_argument("--split", default="run", choices=["run", "seq"])
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    dev = a.device if torch.cuda.is_available() else "cpu"
    takes = load_all_labeled(sensors=a.sensors)
    key = (lambda t: t.run) if a.split == "run" else (lambda t: t.seq)
    groups = sorted({key(t) for t in takes})
    print(f"[contact] {len(takes)} takes; split=leave-one-{a.split}-out groups={groups}; "
          f"dev={dev}; sensors={a.sensors}; ckpt={a.ckpt}", flush=True)

    rows = []
    for g in groups:
        tr = [t for t in takes if key(t) != g]
        te = [t for t in takes if key(t) == g]
        model = finetune_contact(fresh_model(a.ckpt, dev), tr, dev, a.epochs, a.lr, a.batch,
                                 a.win, a.contact_weight)
        fold = evaluate(model, te, dev, a.threshold)
        print(f"  [fold {a.split}{g}] contact_F1={np.mean([x['contact_f1'] for x in fold]):.3f} "
              f"rigid={np.mean([x['rigid_mpjpe'] for x in fold]):.1f}mm (n={len(fold)})", flush=True)
        rows += fold

    # Fold models differ, so summarize the cross-validated rows by macro-average.
    by_motion = collections.defaultdict(list)
    for r in rows:
        by_motion["ALL"].append(r)
        by_motion[r["motion"]].append(r)
    summary = {}
    for k, rs in by_motion.items():
        summary[k] = {
            "contact_precision": float(np.mean([r["contact_precision"] for r in rs])),
            "contact_recall": float(np.mean([r["contact_recall"] for r in rs])),
            "contact_f1": float(np.mean([r["contact_f1"] for r in rs])),
            "contact_accuracy": float(np.mean([r["contact_accuracy"] for r in rs])),
            "rigid_mpjpe": float(np.mean([r["rigid_mpjpe"] for r in rs])),
            "pa_mpjpe": float(np.mean([r["pa_mpjpe"] for r in rs])),
            "n_takes": len(rs),
        }

    tag = a.tag or f"contact_{a.sensors}_{a.split}"
    out_dir = config.RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\n[contact] ALL F1={summary['ALL']['contact_f1']:.3f} acc={summary['ALL']['contact_accuracy']:.3f}")
    print(f"[contact] wrote {out_dir / f'{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
