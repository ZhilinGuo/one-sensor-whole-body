"""Pretrain + cross-validate the MobilePoser-adapted two-stage baseline.

Protocol matches the IMUPoser-adapted baseline exactly (same synthetic AMASS
data, same 150-frame windows, same epochs/optimizer, same aligned FK loss and
metrics), so the two baselines differ only in architecture family:

    # 1) synthetic pretraining (writes results/baselines/<name>/ckpt.pt)
    python mobileposer.py pretrain --name mobileposer_headfeet_all3

    # 2) leave-one-{run,seq}-out finetune + eval (writes results/eval/cv_<tag>.json)
    python mobileposer.py cv --ckpt mobileposer_headfeet_all3 --sensors head --tag mp_head_all3

Pretraining follows MobilePoser's per-module objectives (MSE joints + temporal
smoothness; teacher-forced poser on noise-injected GT joints with MSE 6D pose +
jerk + FK position loss). Real fine-tuning uses the shared aligned FK loss on
the end-to-end pose output plus an aligned auxiliary loss on the joints stage.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from oswb import config
from oswb import jointset
from oswb.model import ACC_SCALE, LEGS_JOINTS, N_INPUT
from oswb.mobileposer import (
    JOINT_NOISE_STD, MobilePoserAdapted, TEMPORAL_WEIGHT, jerk_loss, temporal_loss,
)
from oswb.synth_dataset import WIN, STRIDE
from oswb import fk

from oswb.dataset import RealWindowDataset, load_all_labeled
from oswb.cv import (
    _solve_rot_scale, aligned_joint_loss, evaluate, global_mean_pose, summarize,
)
from oswb import rotation as rot

JS = jointset.get("lower")


class SynthIMUJointsDataset(Dataset):
    """Synthetic windows with IMU input, 6D leg pose, and root-relative joints."""

    def __init__(self, datasets=None, win=WIN, stride=STRIDE):
        datasets = datasets or config.AMASS_DATASETS
        self.win = win
        self.X, self.Y, self.J, self.index = [], [], [], []
        for ds in datasets:
            ddir = config.SYNTH_DIR / "AMASS" / ds
            if not (ddir / "vacc.pt").exists():
                continue
            poses = torch.load(ddir / "pose.pt", weights_only=False)
            vrots = torch.load(ddir / "vrot.pt", weights_only=False)
            vaccs = torch.load(ddir / "vacc.pt", weights_only=False)
            joints = torch.load(ddir / "joint.pt", weights_only=False)
            for i in range(len(poses)):
                T = poses[i].shape[0]
                if T < win // 2:
                    continue
                acc = (vaccs[i].reshape(T, -1) / ACC_SCALE)
                ori = vrots[i].reshape(T, -1)
                X = torch.cat([acc, ori], dim=-1).float()
                mats = rot.axis_angle_to_matrix_torch(poses[i])
                Y = rot.matrix_to_rotation_6d_torch(mats[:, LEGS_JOINTS]).reshape(T, -1).float()
                Jr = (joints[i] - joints[i][:, :1])[:, LEGS_JOINTS].reshape(T, -1).float()
                s = len(self.X)
                self.X.append(X); self.Y.append(Y); self.J.append(Jr)
                for st in range(0, max(1, T - win + 1), stride):
                    self.index.append((s, st))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, st = self.index[idx]
        xw = self.X[s][st:st + self.win]
        yw = self.Y[s][st:st + self.win]
        jw = self.J[s][st:st + self.win]
        if xw.shape[0] < self.win:
            pad = self.win - xw.shape[0]
            xw = torch.cat([xw, torch.zeros(pad, xw.shape[1])])
            yw = torch.cat([yw, torch.zeros(pad, yw.shape[1])])
            jw = torch.cat([jw, torch.zeros(pad, jw.shape[1])])
        return xw, yw, jw


def geodesic_deg(pred6, tgt6):
    Rp = rot.rotation_6d_to_matrix_torch(pred6.reshape(-1, 6))
    Rt = rot.rotation_6d_to_matrix_torch(tgt6.reshape(-1, 6))
    rel = Rp.transpose(-1, -2) @ Rt
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1))).mean()


def pretrain(a):
    dev = a.device if torch.cuda.is_available() else "cpu"
    print("[mp-pretrain] building synthetic dataset...", flush=True)
    ds = SynthIMUJointsDataset()
    n_val = max(1, int(0.05 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    print(f"[mp-pretrain] windows: {len(ds)} (train {len(tr)} / val {len(va)})", flush=True)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    model = MobilePoserAdapted(N_INPUT).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    mse = torch.nn.MSELoss()

    out_dir = config.RESULTS_DIR / "baselines" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best, hist = float("inf"), []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for X, Y, J in tl:
            X, Y, J = X.to(dev), Y.to(dev), J.to(dev)
            B, T = X.shape[0], X.shape[1]
            opt.zero_grad()
            # Joints stage (MobilePoser Joints objective).
            j_pred = model.predict_joints(X)
            loss_j = mse(j_pred, J) + TEMPORAL_WEIGHT * temporal_loss(j_pred)
            # Poser stage, teacher-forced with noised GT joints (MobilePoser Poser).
            j_noisy = J + torch.randn_like(J) * JOINT_NOISE_STD
            pose = model(X, joints=j_noisy)
            loss_p = mse(pose, Y) + TEMPORAL_WEIGHT * jerk_loss(pose)
            fk_pos = fk.legs6d_to_joints(pose.reshape(B * T, 9, 6), device=dev).reshape(B, T, -1)
            loss_p = loss_p + mse(fk_pos, J)
            loss = loss_j + loss_p
            loss.backward()
            opt.step()
            tot += loss.item() * B
        tr_loss = tot / len(tr)

        model.eval()
        vtot, gtot, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, Y, J in vl:
                X, Y = X.to(dev), Y.to(dev)
                pose = model(X)          # end-to-end (predicted joints)
                vtot += mse(pose, Y).item() * X.shape[0]
                gtot += geodesic_deg(pose, Y).item() * X.shape[0]
                n += X.shape[0]
        v_loss, v_geo = vtot / n, gtot / n
        hist.append({"epoch": ep, "train_loss": tr_loss, "val_mse": v_loss, "val_geo_deg": v_geo})
        print(f"[ep {ep:3d}] train {tr_loss:.4f} | val_mse {v_loss:.4f} | "
              f"val_geo {v_geo:.2f}deg | {time.time()-t0:.0f}s", flush=True)
        if v_loss < best:
            best = v_loss
            torch.save({"model": model.state_dict(), "arch": "mobileposer_adapted",
                        "epoch": ep, "val_mse": v_loss, "val_geo_deg": v_geo},
                       out_dir / "ckpt.pt")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"[mp-pretrain] done. best val_mse {best:.4f} -> {out_dir/'ckpt.pt'}", flush=True)


def aligned_points_loss(j_pred, gtj, mask):
    """Aligned (detached rot+scale) L2 on the joints stage, mirroring the FK loss."""
    B, T = j_pred.shape[0], j_pred.shape[1]
    J = j_pred.reshape(B, T, 9, 3)
    Jr, Gr = J - J[:, :, :1], gtj - gtj[:, :, :1]
    w = mask[..., None].expand(-1, -1, 9)
    R, s = _solve_rot_scale(Jr.detach(), Gr, w)
    Ja = s[:, None, None, None] * torch.einsum("bij,btkj->btki", R, Jr)
    err = torch.linalg.norm(Ja - Gr, dim=-1)
    m = mask[..., None]
    return (err * m).sum() / (m.sum() * 9 + 1e-6)


def finetune(model, train_takes, dev, epochs, lr, batch, win):
    ds = RealWindowDataset(train_takes, win=win)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=len(ds) > batch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for X, Yj, M in dl:
            X, Yj, M = X.to(dev), Yj.to(dev), M.to(dev)
            opt.zero_grad()
            j_pred = model.predict_joints(X)
            pose = model(X, joints=j_pred)
            loss = aligned_joint_loss(pose, Yj, M, dev, JS) + aligned_points_loss(j_pred, Yj, M)
            loss.backward()
            opt.step()
    return model


def cv(a):
    dev = a.device if torch.cuda.is_available() else "cpu"

    def fresh_model():
        m = MobilePoserAdapted(N_INPUT).to(dev)
        if a.ckpt.lower() != "none":
            ck = torch.load(config.RESULTS_DIR / "baselines" / a.ckpt / "ckpt.pt",
                            map_location=dev, weights_only=False)
            m.load_state_dict(ck["model"])
        return m

    takes = load_all_labeled(sensors=a.sensors)
    mean_pose = global_mean_pose(takes)
    key = (lambda t: t.run) if a.split == "run" else (lambda t: t.seq)
    groups = sorted({key(t) for t in takes})
    print(f"[mp-cv] {len(takes)} takes; split=leave-one-{a.split}-out groups={groups}; "
          f"dev={dev}; sensors={a.sensors}; ckpt={a.ckpt}", flush=True)

    rows = []
    for g in groups:
        tr = [t for t in takes if key(t) != g]
        te = [t for t in takes if key(t) == g]
        if not te:
            continue
        model = finetune(fresh_model(), tr, dev, a.epochs, a.lr, a.batch, a.win)
        fold = evaluate(model, te, dev, mean_pose, JS)
        print(f"  [fold {a.split}{g}] PA={np.mean([x['pa_mpjpe'] for x in fold]):.1f}mm "
              f"rigid={np.mean([x['rigid_mpjpe'] for x in fold]):.1f}mm (n={len(fold)})", flush=True)
        rows += fold

    tag = a.tag or f"mp_{a.sensors}_{a.split}"
    summ = summarize(rows, tag, list(JS.names))
    out_dir = config.RESULTS_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"cv_{tag}.json").write_text(json.dumps({"rows": rows, "summary": summ}, indent=2))
    print(f"\n[mp-cv] wrote {out_dir / f'cv_{tag}.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain")
    p.add_argument("--name", default="mobileposer_headfeet_all3")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=6)

    c = sub.add_parser("cv")
    c.add_argument("--ckpt", default="mobileposer_headfeet_all3")
    c.add_argument("--device", default="cuda")
    c.add_argument("--epochs", type=int, default=20)
    c.add_argument("--lr", type=float, default=1e-4)
    c.add_argument("--batch", type=int, default=32)
    c.add_argument("--win", type=int, default=150)
    c.add_argument("--tag", default=None)
    c.add_argument("--sensors", default="headfeet",
                   choices=["headfeet", "head", "feet", "lfoot", "rfoot", "none"])
    c.add_argument("--split", default="run", choices=["run", "seq"])

    a = ap.parse_args()
    if a.cmd == "pretrain":
        pretrain(a)
    else:
        cv(a)


if __name__ == "__main__":
    main()
