"""Synthesize virtual head + left-foot + right-foot IMU streams from AMASS.

Mirrors the DIP/IMUPoser synthesis convention so synthetic and real share a
representation, but for our 3-sensor head+feet layout at the benchmark's 30 Hz
rate (matching the 30 fps camera pseudo-GT):

  orientation  : SMPL global joint rotation at [head=15, L-ankle=7, R-ankle=8]
  acceleration : 2nd finite difference of mesh vertices [411, 3327, 6727] * fps^2
  targets      : SMPL pose (24x3 axis-angle), joints (24x3), root pelvis (idx 0)

Output per AMASS sub-dataset:
``data/synthetic/AMASS/<ds>/{pose,shape,tran,joint,vrot,vacc}.pt``.

Requires the AMASS download and the SMPL male model (see INSTALL.md):

    python synth_amass.py                     # all three paper sub-datasets
    python synth_amass.py --datasets CMU --limit 20   # quick debug run
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings

import numpy as np
import torch

from oswb import config
from oswb import rotation as rot
from oswb.smpl import SMPLModel

warnings.filterwarnings("ignore")

TARGET_FPS = 30
VI_MASK = torch.tensor([config.VERT_HEAD, config.VERT_LEFT_FOOT, config.VERT_RIGHT_FOOT])
JI_MASK = torch.tensor([config.JOINT_HEAD, config.JOINT_LEFT_FOOT, config.JOINT_RIGHT_FOOT])
# AMASS (y-up) -> DIP/IMUPoser frame, matching IMUPoser's preprocessing.
AMASS_ROT = torch.tensor([[[1.0, 0, 0], [0, 0, 1], [0, -1, 0]]])


def _syn_acc(v: torch.Tensor, fps: int = TARGET_FPS) -> torch.Tensor:
    """Central 2nd difference of vertex positions -> acceleration (m/s^2)."""
    f2 = float(fps * fps)
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * f2 for i in range(v.shape[0] - 2)])
    return torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))


def _load_amass_seq(npz):
    try:
        c = np.load(npz)
    except Exception:
        return None
    fr = int(round(float(c["mocap_framerate"])))
    if fr % TARGET_FPS != 0:
        return None
    step = fr // TARGET_FPS
    pose = c["poses"][::step, :72].astype(np.float32)
    tran = c["trans"][::step].astype(np.float32)
    beta = c["betas"][:10].astype(np.float32)
    return pose, tran, beta


def process_dataset(ds_name: str, body_model: SMPLModel, limit=None) -> int:
    files = sorted(glob.glob(os.path.join(config.AMASS_DIR, ds_name, "*/*_poses.npz")))
    if limit:
        files = files[:limit]
    out = {k: [] for k in ("pose", "shape", "tran", "joint", "vrot", "vacc")}
    n_ok = 0
    for npz in files:
        seq = _load_amass_seq(npz)
        if seq is None:
            continue
        pose_aa, tran, beta = seq
        if pose_aa.shape[0] <= 12:
            continue
        pose = torch.from_numpy(pose_aa).view(-1, 24, 3)
        tran = torch.from_numpy(tran)
        shape = torch.from_numpy(beta)
        # align global frame (root orient + translation) to the DIP convention
        tran = AMASS_ROT.matmul(tran.unsqueeze(-1)).view_as(tran)
        pose[:, 0] = rot.matrix_to_axis_angle_torch(
            AMASS_ROT.matmul(rot.axis_angle_to_matrix_torch(pose[:, 0]))
        )
        p = rot.axis_angle_to_matrix_torch(pose).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape, tran, calc_mesh=True)
        out["pose"].append(pose.clone())
        out["shape"].append(shape.clone())
        out["tran"].append(tran.clone())
        out["joint"].append(joint[:, :24].contiguous().clone())
        out["vrot"].append(grot[:, JI_MASK].clone())          # N,3,3,3
        out["vacc"].append(_syn_acc(vert[:, VI_MASK]).clone())  # N,3,3
        n_ok += 1

    if n_ok:
        ds_dir = config.SYNTH_DIR / "AMASS" / ds_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            torch.save(v, ds_dir / f"{k}.pt")
    return n_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=config.AMASS_DATASETS)
    ap.add_argument("--limit", type=int, default=None, help="cap seqs/dataset (debug)")
    a = ap.parse_args()
    config.ensure_dirs()
    body_model = SMPLModel(str(config.SMPL_MALE_PKL))
    for ds in a.datasets:
        ds_dir = config.SYNTH_DIR / "AMASS" / ds
        done = all((ds_dir / f"{k}.pt").exists()
                   for k in ("pose", "shape", "tran", "joint", "vrot", "vacc"))
        if done and not a.limit:
            print(f"{ds}: already synthesized, skipping", flush=True)
            continue
        n = process_dataset(ds, body_model, limit=a.limit)
        print(f"{ds}: {n} sequences synthesized -> {ds_dir}", flush=True)


if __name__ == "__main__":
    main()
