"""Windowed synthetic dataset for head(+feet) sparse-IMU pose pretraining.

Reads the per-dataset tensors written by ``synth_amass.py`` and builds
fixed-length windows of model input (global accel/ACC_SCALE + global orientation,
3 sensors -> 36-D) and pose targets (6D rotations). Everything is in the SMPL
global frame, matching IMUPoser's "global" model convention.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from oswb import config
from oswb.model import ACC_SCALE, LEGS_JOINTS
from oswb import rotation as rot

WIN = 150       # 5 s at 30 Hz
STRIDE = 75
FPS = 30.0      # synthetic streams are synthesized on the project 30 Hz grid


def _seq_to_xy(pose_aa, vrot, vacc, joints=LEGS_JOINTS):
    """pose (T,24,3) aa, vrot (T,3,3,3), vacc (T,3,3) -> X (T,36), Y (T,6*J)."""
    T = pose_aa.shape[0]
    acc = (vacc.reshape(T, -1) / ACC_SCALE)            # (T,9)
    ori = vrot.reshape(T, -1)                          # (T,27)
    X = torch.cat([acc, ori], dim=-1).float()          # (T,36)
    mats = rot.axis_angle_to_matrix_torch(pose_aa)     # (T,24,3,3)
    sel = mats[:, joints]                              # (T,J,3,3)
    y6 = rot.matrix_to_rotation_6d_torch(sel)          # (T,J,6)
    Y = y6.reshape(T, -1).float()                      # (T,6*J)
    return X, Y


class SynthIMUDataset(Dataset):
    def __init__(self, datasets=None, win=WIN, stride=STRIDE, max_seqs=None, joints=None):
        datasets = datasets or config.AMASS_DATASETS
        joints = LEGS_JOINTS if joints is None else joints
        self.win = win
        self.X, self.Y, self.index = [], [], []
        for ds in datasets:
            ddir = config.SYNTH_DIR / "AMASS" / ds
            if not (ddir / "vacc.pt").exists():
                continue
            poses = torch.load(ddir / "pose.pt", weights_only=False)
            vrots = torch.load(ddir / "vrot.pt", weights_only=False)
            vaccs = torch.load(ddir / "vacc.pt", weights_only=False)
            for i in range(len(poses)):
                if max_seqs and len(self.X) >= max_seqs:
                    break
                T = poses[i].shape[0]
                if T < win // 2:
                    continue
                X, Y = _seq_to_xy(poses[i], vrots[i], vaccs[i], joints)
                s = len(self.X)
                self.X.append(X)
                self.Y.append(Y)
                for st in range(0, max(1, T - win + 1), stride):
                    self.index.append((s, st))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, st = self.index[idx]
        X, Y = self.X[s], self.Y[s]
        xw, yw = X[st:st + self.win], Y[st:st + self.win]
        if xw.shape[0] < self.win:  # pad the tail window
            padx = torch.zeros(self.win - xw.shape[0], xw.shape[1])
            pady = torch.zeros(self.win - yw.shape[0], yw.shape[1])
            xw, yw = torch.cat([xw, padx]), torch.cat([yw, pady])
        return xw, yw
