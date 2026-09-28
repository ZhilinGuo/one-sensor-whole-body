"""Selectable model joint sets: the 9-joint lower-body set and a 20-joint
"full-body" set used for the head-only full-body extension.

The full set predicts every SMPL body joint except the collars (13,14) and hand
stubs (22,23). Seventeen of the twenty have direct MHR70 pseudo-GT counterparts
and are supervised/evaluated; spine1/2/3 (SMPL 3,6,9) carry loss weight 0 --
they remain in the output for forward-kinematics capacity (shoulder/elbow/wrist
positions depend on torso bend) and receive gradients only through their
supervised descendants. Their target slots are filled with a pelvis-to-neck
interpolation so tensors stay finite; they are excluded from all metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oswb.mhr70 import MHR70_NAME_TO_IDX

# SMPL indices predicted for each set (order = model output order).
LOWER_SMPL = [0, 1, 2, 4, 5, 7, 8, 10, 11]
FULL_SMPL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]

LOWER_NAMES = [
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_foot", "right_foot",
]
FULL_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "head", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist",
]

_SPINE = {"spine1", "spine2", "spine3"}

# Joint groups for reporting (indices into the FULL output order).
FULL_LOWER_IDX = [i for i, n in enumerate(FULL_NAMES) if n in LOWER_NAMES]
FULL_UPPER_IDX = [i for i, n in enumerate(FULL_NAMES)
                  if n in ("neck", "head", "left_shoulder", "right_shoulder",
                           "left_elbow", "right_elbow", "left_wrist", "right_wrist")]


@dataclass(frozen=True)
class JointSet:
    name: str
    smpl: list[int]          # SMPL joint indices, in model-output order
    names: list[str]         # joint names, same order
    supw: list[float]        # per-joint supervision weight (0 = unsupervised)

    @property
    def n_joints(self) -> int:
        return len(self.smpl)

    @property
    def n_output(self) -> int:
        return 6 * len(self.smpl)

    @property
    def sup_idx(self) -> list[int]:
        """Output indices that receive loss (weight > 0)."""
        return [i for i, w in enumerate(self.supw) if w > 0]

    @property
    def eval_idx(self) -> list[int]:
        """Supervised indices with real (non-interpolated) labels, i.e. evaluated.

        Spine slots supervised only to constrain the FK torso chain (weight
        0.25, interpolated targets) are excluded from all metrics.
        """
        return [i for i, w in enumerate(self.supw) if w >= 0.5]

    @property
    def groups(self) -> dict[str, list[int]]:
        """Named joint groups (indices into output order) for reporting."""
        if self.name.startswith("full"):
            return {"lower": FULL_LOWER_IDX, "upper": FULL_UPPER_IDX}
        return {"lower": list(range(self.n_joints))}

    def loss_weights(self) -> list[float] | None:
        """Per-supervised-joint fine-tune loss weights (None = uniform).

        Full sets carry their weights directly in ``supw``: lower 1.0, upper 0.5
        (single-view arm pseudo-GT is the noisiest label source), spines 0.25
        when supervised at all (``fulls`` set).
        """
        if not self.name.startswith("full"):
            return None
        return [self.supw[i] for i in self.sup_idx]

    def align_idx(self) -> list[int] | None:
        """Supervised-joint positions used to solve the training alignment.

        On the full set the per-window rot+scale is solved on the lower-body
        joints only (the reliable labels) and applied to all joints; aligning on
        noisy arm labels corrupts the whole window's training signal.
        """
        if not self.name.startswith("full"):
            return None
        lower = set(FULL_LOWER_IDX)
        return [p for p, out_i in enumerate(self.sup_idx) if out_i in lower]


_JOINT_SETS = {
    "lower": JointSet("lower", LOWER_SMPL, LOWER_NAMES, [1.0] * len(LOWER_SMPL)),
    # full: spines unsupervised (weight 0) — FK capacity only.
    "full": JointSet("full", FULL_SMPL, FULL_NAMES,
                     [0.0 if n in _SPINE else (0.5 if n in {FULL_NAMES[i] for i in FULL_UPPER_IDX} else 1.0)
                      for n in FULL_NAMES]),
    # fulls: spines weakly supervised (0.25) with interpolated targets so the
    # torso FK chain is not unconstrained; spines still excluded from metrics.
    "fulls": JointSet("fulls", FULL_SMPL, FULL_NAMES,
                      [0.25 if n in _SPINE else (0.5 if n in {FULL_NAMES[i] for i in FULL_UPPER_IDX} else 1.0)
                       for n in FULL_NAMES]),
}


def get(name: str) -> JointSet:
    return _JOINT_SETS[name]


def full_body_from_mhr70(kp70: np.ndarray) -> np.ndarray:
    """Map ``(..., 70, 3)`` MHR70 keypoints to ``(..., 20, 3)`` full-set targets.

    Supervised joints use their direct MHR70 counterpart; the unsupervised spine
    slots are pelvis-to-neck interpolations (never supervised, never evaluated).
    ``head`` is the ear midpoint (the earbuds sit at the ears; more stable than
    the nose under single-view occlusion).
    """
    kp70 = np.asarray(kp70, dtype=np.float64)
    g = lambda n: kp70[..., MHR70_NAME_TO_IDX[n], :]  # noqa: E731
    pelvis = 0.5 * (g("left_hip") + g("right_hip"))
    neck = g("neck")
    lerp = lambda a: pelvis + a * (neck - pelvis)  # noqa: E731

    by_name = {
        "pelvis": pelvis,
        "left_hip": g("left_hip"), "right_hip": g("right_hip"),
        "spine1": lerp(0.25), "spine2": lerp(0.5), "spine3": lerp(0.75),
        "left_knee": g("left_knee"), "right_knee": g("right_knee"),
        "left_ankle": g("left_ankle"), "right_ankle": g("right_ankle"),
        "left_foot": g("left_big_toe"), "right_foot": g("right_big_toe"),
        "neck": neck,
        "head": 0.5 * (g("left_ear") + g("right_ear")),
        "left_shoulder": g("left_shoulder"), "right_shoulder": g("right_shoulder"),
        "left_elbow": g("left_elbow"), "right_elbow": g("right_elbow"),
        "left_wrist": g("left_wrist"), "right_wrist": g("right_wrist"),
    }
    return np.stack([by_name[n] for n in FULL_NAMES], axis=-2)
