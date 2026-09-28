"""Real captured takes -> model inputs + joint targets.

Calibrates each take's IMUs into the common gravity world (``oswb.calib``) and
packs the per-sensor accel+orientation into the exact 36-D layout the synthetic
pretraining uses, so a pretrained model can be evaluated/fine-tuned directly.
Targets are the pseudo-GT joints with a validity mask. Leave-one-run-out and
leave-one-motion-out splits are grouped by capture run / motion script.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from oswb import config
from oswb import jointset
from oswb.calib import calibrate_take
from oswb.model import ACC_SCALE
from oswb.schema import Take


@dataclass
class TakeArrays:
    run: int
    seq: int
    motion_class: str
    fps: float
    X: np.ndarray          # (T,36) model input
    Yj: np.ndarray         # (T,J,3) gt joints for the active joint set (camera frame, m)
    valid: np.ndarray      # (T,) bool
    contacts: np.ndarray   # (T,2) bool or None
    calib: dict


_SENSOR_SLICES = {"head": 0, "lfoot": 1, "rfoot": 2}
_SUBSETS = {"headfeet": ("head", "lfoot", "rfoot"),
            "head": ("head",), "feet": ("lfoot", "rfoot"),
            "lfoot": ("lfoot",), "rfoot": ("rfoot",),
            "none": ()}  # zero-input control: measures the motion-prior floor


def _pack(acc, ori) -> np.ndarray:
    T = acc.shape[0]
    return np.concatenate([(acc / ACC_SCALE).reshape(T, -1),
                           ori.reshape(T, -1)], axis=-1).astype(np.float32)


def _pack_input(cal) -> np.ndarray:
    acc = np.stack([cal["head_a"], cal["lfoot_a"], cal["rfoot_a"]], axis=1)
    ori = np.stack([cal["head_R"], cal["lfoot_R"], cal["rfoot_R"]], axis=1)
    return _pack(acc, ori)


def _raw_input(take: Take) -> np.ndarray:
    """No-calibration baseline: raw device-frame orientation + acceleration."""
    acc = np.stack([np.asarray(take.imu_head_acc, float),
                    np.asarray(take.imu_lfoot_acc, float),
                    np.asarray(take.imu_rfoot_acc, float)], axis=1)
    ori = np.stack([np.asarray(take.imu_head_R, float),
                    np.asarray(take.imu_lfoot_R, float),
                    np.asarray(take.imu_rfoot_R, float)], axis=1)
    return _pack(acc, ori)


def _mask_sensors(X: np.ndarray, sensors: str) -> np.ndarray:
    """Zero the accel(3)+ori(9) channels of any sensor not in the subset."""
    keep = set(_SUBSETS[sensors])
    X = X.copy()
    for name, s in _SENSOR_SLICES.items():
        if name in keep:
            continue
        X[:, s * 3:(s + 1) * 3] = 0.0                       # accel block
        X[:, 9 + s * 9:9 + (s + 1) * 9] = 0.0               # ori block
    return X


def load_take(take: Take, calib_mode: str = "full", sensors: str = "headfeet",
              joints: str = "lower") -> TakeArrays:
    if calib_mode == "raw":
        X, cal = _raw_input(take), {}
    else:
        c = calibrate_take(take)
        X, cal = _pack_input(c), c["calib"]
    if sensors != "headfeet":
        X = _mask_sensors(X, sensors)
    if joints.startswith("full"):
        Yj = jointset.full_body_from_mhr70(np.asarray(take.gt_joints_mhr70)).astype(np.float32)
    else:
        Yj = np.asarray(take.gt_joints_lower, np.float32)
    valid = np.asarray(take.gt_valid, bool) if take.gt_valid is not None \
        else np.ones(len(X), bool)
    return TakeArrays(
        run=take.run, seq=take.seq, motion_class=take.motion_class, fps=float(take.fps),
        X=X, Yj=Yj, valid=valid,
        contacts=np.asarray(take.gt_contacts, bool) if take.gt_contacts is not None else None,
        calib=cal,
    )


def load_all_labeled(calib_mode: str = "full", sensors: str = "headfeet",
                     joints: str = "lower"):
    takes = [t for t in (Take.load(p) for p in sorted(config.PROCESSED_DIR.glob("run*_seq*.npz")))
             if t.has_labels]
    return [load_take(t, calib_mode=calib_mode, sensors=sensors, joints=joints)
            for t in takes]


class RealWindowDataset(Dataset):
    """Fixed-length windows over a list of ``TakeArrays`` (for fine-tuning)."""

    def __init__(self, takes, win=150, stride=75):
        self.win = win
        self.Xs, self.Ys, self.Ms, self.index = [], [], [], []
        for k, ta in enumerate(takes):
            T = ta.X.shape[0]
            self.Xs.append(torch.from_numpy(ta.X))
            self.Ys.append(torch.from_numpy(ta.Yj))
            self.Ms.append(torch.from_numpy(ta.valid.astype(np.float32)))
            for st in range(0, max(1, T - win + 1), stride):
                self.index.append((k, st))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        k, st = self.index[idx]
        x, y, m = self.Xs[k][st:st + self.win], self.Ys[k][st:st + self.win], self.Ms[k][st:st + self.win]
        if x.shape[0] < self.win:
            pad = self.win - x.shape[0]
            x = torch.cat([x, torch.zeros(pad, x.shape[1])])
            y = torch.cat([y, torch.zeros(pad, *y.shape[1:])])
            m = torch.cat([m, torch.zeros(pad)])
        return x, y, m
