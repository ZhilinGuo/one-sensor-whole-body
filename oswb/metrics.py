"""Pose-estimation metrics for the head(+feet) -> body-pose task.

All position metrics take ``(T, J, 3)`` arrays in metres and return millimetres
unless noted. Rotation metrics take ``(T, J, 3, 3)`` matrices and return degrees.
Single source of truth shared by every model so numbers are directly comparable.
"""
from __future__ import annotations

import numpy as np

M2MM = 1000.0


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def root_align(joints: np.ndarray, root_idx: int = 0) -> np.ndarray:
    """Subtract the root joint per frame -> root-relative joints."""
    return joints - joints[:, root_idx : root_idx + 1, :]


def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-frame similarity (Umeyama) alignment of ``pred`` onto ``gt``."""
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    out = np.empty_like(pred)
    for i in range(pred.shape[0]):
        out[i] = _umeyama(pred[i], gt[i])
    return out


def _umeyama(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    mu_x, mu_y = X.mean(0), Y.mean(0)
    X0, Y0 = X - mu_x, Y - mu_y
    cov = Y0.T @ X0 / X.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_x = (X0**2).sum() / X.shape[0]
    scale = np.trace(np.diag(D) @ S) / (var_x + 1e-12)
    return (scale * (R @ X0.T).T) + mu_y


# --------------------------------------------------------------------------- #
# Position metrics (mm)
# --------------------------------------------------------------------------- #
def mpjpe(pred: np.ndarray, gt: np.ndarray, align: str = "root", root_idx: int = 0):
    """Mean per-joint position error (mm). ``align`` in {none, root, procrustes}."""
    p, g = _apply_align(pred, gt, align, root_idx)
    return float(np.linalg.norm(p - g, axis=-1).mean() * M2MM)


def pa_mpjpe(pred, gt):
    """Procrustes-aligned MPJPE (mm)."""
    return mpjpe(pred, gt, align="procrustes")


def mpjve(pred, gt, fps: float, align: str = "root", root_idx: int = 0):
    """Mean per-joint velocity error (mm/s) from finite differences."""
    p, g = _apply_align(pred, gt, align, root_idx)
    vp = np.diff(p, axis=0) * fps
    vg = np.diff(g, axis=0) * fps
    return float(np.linalg.norm(vp - vg, axis=-1).mean() * M2MM)


def jitter(joints: np.ndarray, fps: float) -> float:
    """Mean jerk magnitude (reported in km/s^3 as in TransPose)."""
    j = np.asarray(joints, float)
    if j.shape[0] < 4:
        return float("nan")
    jerk = np.diff(j, n=3, axis=0) * (fps**3)
    return float(np.linalg.norm(jerk, axis=-1).mean() / 1000.0)


def _apply_align(pred, gt, align, root_idx):
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    if align == "none":
        return pred, gt
    if align == "root":
        return root_align(pred, root_idx), root_align(gt, root_idx)
    if align == "procrustes":
        return procrustes_align(pred, gt), gt
    raise ValueError(f"unknown align: {align}")


# --------------------------------------------------------------------------- #
# Contact
# --------------------------------------------------------------------------- #
def contact_prf1(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Precision / recall / F1 / accuracy for boolean contact arrays (any shape)."""
    pred = np.asarray(pred).astype(bool).ravel()
    gt = np.asarray(gt).astype(bool).ravel()
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    tn = int((~pred & ~gt).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(pred), 1)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
