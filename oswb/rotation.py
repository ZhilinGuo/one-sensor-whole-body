"""Rotation conversions used by the model, synthesis, and evaluation code.

Torch-first: the training and evaluation paths are torch. The 6D rotation
representation follows Zhou et al. 2019 (first two matrix columns).
"""
from __future__ import annotations

import numpy as np


def matrix_to_rotation_6d_torch(mats):
    """(..., 3, 3) -> (..., 6): first two columns, column-major flatten."""
    import torch  # noqa: F401

    return mats[..., :, :2].transpose(-1, -2).reshape(*mats.shape[:-2], 6).contiguous()


def rotation_6d_to_matrix_torch(d6):
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt."""
    import torch

    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def axis_angle_to_matrix_torch(aa):
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices (Rodrigues, torch)."""
    import torch

    theta = torch.norm(aa, dim=-1, keepdim=True)
    k = aa / (theta + 1e-8)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1)
    K = K.reshape(*k.shape[:-1], 3, 3)
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def matrix_to_axis_angle_torch(mats):
    """(..., 3, 3) rotation matrices -> (..., 3) axis-angle (Rodrigues inverse).

    The angle uses the well-conditioned atan2(|skew|, trace) form -- arccos of
    the trace is ill-conditioned near pi (dtheta/dcos ~ 1/sin(theta)), which
    corrupts the root orientation of AMASS sequences that face "backwards".
    The axis is the normalized skew vector; only at exactly theta = pi (skew
    vector vanishes) is it recovered from the symmetric part instead.
    Computed in float64.
    """
    import torch

    dtype = mats.dtype
    m = mats.double()
    tr = m.diagonal(dim1=-2, dim2=-1).sum(-1)
    c = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0)
    w = torch.stack([m[..., 2, 1] - m[..., 1, 2],
                     m[..., 0, 2] - m[..., 2, 0],
                     m[..., 1, 0] - m[..., 0, 1]], dim=-1)
    wn = w.norm(dim=-1)                                # = 2 sin(theta)
    theta = torch.atan2(wn / 2.0, c)

    eps = 1e-12
    use_skew = wn > eps
    axis = w / wn[..., None].clamp(min=eps)

    if (~use_skew).any():
        # theta ~ 0: axis irrelevant (zero rotation). theta ~ pi: recover the
        # axis from uu^T = ((R + R^T)/2 - cos I)/(1 - cos), anchoring on the
        # largest diagonal entry; any valid sign round-trips to the same matrix.
        S = (m + m.transpose(-1, -2)) / 2.0
        eye = torch.eye(3, dtype=torch.float64, device=m.device)
        uu = (S - c[..., None, None] * eye) / (1.0 - c[..., None, None]).clamp(min=eps)
        diag = uu.diagonal(dim1=-2, dim2=-1).clamp(min=0.0)      # (...,3) = u_i^2
        anchor = diag.argmax(dim=-1)                             # (...,)
        u_anchor = diag.gather(-1, anchor[..., None]).sqrt().clamp(min=eps)
        col = uu.gather(-1, anchor[..., None, None].expand(*uu.shape[:-1], 1)).squeeze(-1)
        u_pi = col / u_anchor
        is_pi = (~use_skew) & (c < 0)
        axis = torch.where(is_pi[..., None], u_pi, axis)
        axis = torch.where((~use_skew & ~is_pi)[..., None], torch.zeros_like(axis), axis)

    return (axis * theta[..., None]).to(dtype)


def world_angular_velocity(mats: np.ndarray, fps: float) -> np.ndarray:
    """World-frame angular velocity (rad/s) from a rotation sequence.

    ``omega_t = log(R_{t+1} R_t^T) * fps``. Returns (T, 3).
    """
    from scipy.spatial.transform import Rotation as R

    mats = np.asarray(mats, dtype=np.float64)
    T = mats.shape[0]
    omega = np.zeros((T, 3))
    if T < 2:
        return omega
    rel = np.matmul(mats[1:], np.transpose(mats[:-1], (0, 2, 1)))
    w = R.from_matrix(rel).as_rotvec() * float(fps)
    omega[:-1] = w
    omega[-1] = w[-1]
    return omega
