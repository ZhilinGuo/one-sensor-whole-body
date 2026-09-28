"""SMPL forward kinematics: predicted 6D rotations -> 3D joint positions.

Uses the project's minimal SMPL implementation (``oswb.smpl``) so predicted 6D
rotations map to 3D joint positions in the exact convention the model was
trained on. Non-predicted joints are held at identity (they do not affect the
lower-body kinematic chain, so lower-body joint positions are exact).
"""
from __future__ import annotations

import torch

from oswb import config
from oswb.model import LEGS_JOINTS
from oswb.smpl import SMPLModel
from oswb import rotation as rot

_MODELS: dict[str, SMPLModel] = {}


def _model(device="cpu") -> SMPLModel:
    key = str(device)
    if key not in _MODELS:
        _MODELS[key] = SMPLModel(str(config.SMPL_MALE_PKL), device=torch.device(device))
    return _MODELS[key]


def sixd_to_joints(pred6: torch.Tensor, joints, device=None) -> torch.Tensor:
    """``(N,J,6)`` 6D rotations for SMPL ``joints`` -> ``(N,J,3)`` positions (m).

    Non-predicted joints are held at identity; predicted joints off the lower
    chain (e.g. spine) propagate to their descendants through FK as usual.
    """
    device = device if device is not None else pred6.device
    model = _model(device)
    N = pred6.shape[0]
    J = len(joints)
    mats = rot.rotation_6d_to_matrix_torch(pred6.reshape(N, J, 6).to(device))  # (N,J,3,3)
    full = torch.eye(3, device=mats.device).repeat(N, 24, 1, 1)
    idx = torch.tensor(joints, device=mats.device)
    full[:, idx] = mats.to(full.dtype)
    _, out = model.forward_kinematics(full)                          # (N,24,3)
    return out[:, idx]


def legs6d_to_joints(pred6: torch.Tensor, device=None) -> torch.Tensor:
    """``(N,9,6)`` 6D leg rotations -> ``(N,9,3)`` lower-body joint positions (m)."""
    return sixd_to_joints(pred6, LEGS_JOINTS, device=device)
