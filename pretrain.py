"""Pretrain the sparse-IMU pose model on synthetic AMASS (IMUPoser-adapted).

Plain-torch training of ``PoseRNN`` (IMUPoser BiLSTM) with an L1 loss on the 6D
pose plus a geodesic monitor. Saves to ``results/baselines/<name>/ckpt.pt``.

    python pretrain.py --name imuposer_headfeet_all3
    python pretrain.py --name imuposer_causal_all3 --causal      # streaming variant
    python pretrain.py --name imuposer_full20_all3 --joints full # full-body extension
"""
from __future__ import annotations

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader, random_split

from oswb import config
from oswb import jointset
from oswb.model import PoseRNN, N_INPUT
from oswb.synth_dataset import SynthIMUDataset
from oswb import rotation as rot


def geodesic_deg(pred6, tgt6):
    Rp = rot.rotation_6d_to_matrix_torch(pred6.reshape(-1, 6))
    Rt = rot.rotation_6d_to_matrix_torch(tgt6.reshape(-1, 6))
    rel = Rp.transpose(-1, -2) @ Rt
    tr = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.arccos(cos)).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="imuposer_headfeet_all3")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seqs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--joints", choices=["lower", "full"], default="lower",
                    help="model joint set (full = 20-joint full-body extension)")
    ap.add_argument("--causal", action="store_true",
                    help="unidirectional LSTM (streaming variant)")
    a = ap.parse_args()

    js = jointset.get(a.joints)
    dev = a.device if torch.cuda.is_available() else "cpu"
    print(f"[train] building synthetic dataset (joints={a.joints}, J={js.n_joints})...", flush=True)
    ds = SynthIMUDataset(datasets=a.datasets, max_seqs=a.max_seqs, joints=js.smpl)
    n_val = max(1, int(0.05 * len(ds)))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    print(f"[train] windows: {len(ds)} (train {len(tr)} / val {len(va)})", flush=True)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True)
    vl = DataLoader(va, batch_size=a.batch, shuffle=False, num_workers=a.workers)

    model = PoseRNN(N_INPUT, js.n_output, bidirectional=not a.causal).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = torch.nn.L1Loss()

    out_dir = config.RESULTS_DIR / "baselines" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    hist = []
    for ep in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for X, Y in tl:
            X, Y = X.to(dev), Y.to(dev)
            opt.zero_grad()
            loss = lossf(model(X), Y)
            loss.backward()
            opt.step()
            tot += loss.item() * X.shape[0]
        tr_loss = tot / len(tr)

        model.eval()
        vtot, gtot, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, Y in vl:
                X, Y = X.to(dev), Y.to(dev)
                pred = model(X)
                vtot += lossf(pred, Y).item() * X.shape[0]
                gtot += geodesic_deg(pred, Y).item() * X.shape[0]
                n += X.shape[0]
        v_loss, v_geo = vtot / n, gtot / n
        hist.append({"epoch": ep, "train_l1": tr_loss, "val_l1": v_loss, "val_geo_deg": v_geo})
        print(f"[ep {ep:3d}] train_l1 {tr_loss:.4f} | val_l1 {v_loss:.4f} | "
              f"val_geo {v_geo:.2f}deg | {time.time()-t0:.0f}s", flush=True)
        if v_loss < best:
            best = v_loss
            torch.save({"model": model.state_dict(), "n_input": N_INPUT,
                        "n_output": js.n_output, "n_out_joints": js.n_joints,
                        "joints": a.joints, "arch": "plain",
                        "causal": a.causal,
                        "epoch": ep, "val_l1": v_loss, "val_geo_deg": v_geo},
                       out_dir / "ckpt.pt")
        (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    print(f"[train] done. best val_l1 {best:.4f} -> {out_dir/'ckpt.pt'}", flush=True)


if __name__ == "__main__":
    main()
