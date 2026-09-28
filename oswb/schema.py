"""Canonical per-take benchmark representation.

One ``.npz`` per take under ``data/processed/run{R}_seq{S}.npz``. Every
downstream consumer -- training, evaluation, ablations -- reads this single
contract so inputs/targets are unambiguous.

Conventions
-----------
* Time ``t`` is seconds on a uniform 30 Hz grid (``fps``).
* IMU orientation ``*_R`` are (T,3,3) rotation matrices in each sensor's
  native gravity-referenced frame; a constant per-sensor calibration to the
  common world frame is estimated at load time (``oswb.calib``).
* IMU acceleration is m/s^2. Head accel is gravity-free; foot accel includes
  gravity unless ``meta['foot_acc_gravity_removed']`` is set.
* Feet have no gyroscope (``*_gyro`` exists for the head only).
* Pseudo-GT joints are metres in the camera frame; ``gt_valid`` flags frames
  with a confident detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from oswb import config
from oswb import io


@dataclass
class Take:
    run: int
    seq: int
    fps: float = 30.0
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    t0_unix: float = 0.0

    # IMU streams (resampled to t)
    imu_head_R: np.ndarray | None = None
    imu_head_acc: np.ndarray | None = None
    imu_head_gyro: np.ndarray | None = None
    imu_lfoot_R: np.ndarray | None = None
    imu_lfoot_acc: np.ndarray | None = None
    imu_rfoot_R: np.ndarray | None = None
    imu_rfoot_acc: np.ndarray | None = None

    # Pseudo-GT (resampled to t)
    gt_joints_lower: np.ndarray | None = None   # (T, 9, 3)
    gt_joints_mhr70: np.ndarray | None = None   # (T, 70, 3)
    gt_contacts: np.ndarray | None = None       # (T, 2) bool
    gt_valid: np.ndarray | None = None          # (T,) bool

    meta: dict = field(default_factory=dict)

    @property
    def motion_class(self) -> str:
        return config.SEQ_MOTION_CLASS.get(self.seq, "unknown")

    @property
    def n_frames(self) -> int:
        return int(len(self.t))

    @property
    def has_labels(self) -> bool:
        return self.gt_joints_lower is not None

    @classmethod
    def load(cls, path: str | Path) -> "Take":
        d = io.load_npz(path)
        kwargs = {}
        valid = {f for f in cls.__dataclass_fields__}
        for k, v in d.items():
            if k not in valid:
                continue
            if k in ("run", "seq"):
                kwargs[k] = int(v)
            elif k in ("fps", "t0_unix"):
                kwargs[k] = float(v)
            elif k == "meta":
                kwargs[k] = v if isinstance(v, dict) else {}
            else:
                kwargs[k] = v
        return cls(**kwargs)

    def summary(self) -> str:
        bits = [f"run{self.run}_seq{self.seq} [{self.motion_class}] "
                f"T={self.n_frames} @{self.fps:g}Hz"]
        bits.append("GT:lower" + str(self.gt_joints_lower.shape)
                    if self.has_labels else "GT:none")
        return " | ".join(bits)


def iter_processed():
    """Yield ``Take`` for every processed ``.npz`` present, sorted by (run, seq)."""
    for p in sorted(config.PROCESSED_DIR.glob("run*_seq*.npz")):
        yield Take.load(p)
