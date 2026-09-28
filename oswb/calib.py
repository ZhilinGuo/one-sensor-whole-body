"""Calibrate consumer IMUs (earbud head, insole feet) into the SMPL world frame.

Each device reports orientation in its own gravity-referenced world frame (a per
recording yaw offset) with an unknown, constant sensor->bone mounting rotation.
Synthetic training data is in the SMPL world frame (gravity down). To make real
inputs match that distribution we estimate, per take and per sensor, two constant
rotations ``A`` (device-world -> common gravity world) and ``B`` (sensor -> bone)
by aligning the IMU orientation stream to keypoint-derived bone frames via a
two-sided orthogonal Procrustes fit:

    minimize_{A,B in SO(3)}  sum_t || R_bone_world(t) - A R_imu(t) B ||_F^2 .

The residual convention gap between the keypoint bone frame and the exact SMPL
joint frame is a single global constant per bone, absorbed by fine-tuning.
"""
from __future__ import annotations

import numpy as np

from oswb import mhr70


def _normalize(v, axis=-1, eps=1e-8):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def _frame_from_two(primary, secondary):
    """Right-handed orthonormal frame (T,3,3) with columns [e0,e1,e2]."""
    e0 = _normalize(primary)
    e2 = _normalize(np.cross(e0, secondary))
    e1 = np.cross(e2, e0)
    return np.stack([e0, e1, e2], axis=-1)


def head_frame_cam(kp70):
    """(T,70,3) -> (T,3,3) head bone frame in camera coords.

    Lateral axis right_ear->left_ear; forward axis ear-midpoint->nose.
    """
    le = kp70[:, mhr70.MHR70_NAME_TO_IDX["left_ear"], :]
    re = kp70[:, mhr70.MHR70_NAME_TO_IDX["right_ear"], :]
    nose = kp70[:, mhr70.MHR70_NAME_TO_IDX["nose"], :]
    lateral = le - re
    forward = nose - 0.5 * (le + re)
    return _frame_from_two(lateral, forward)


def foot_frame_cam(kp70, side):
    """(T,70,3) -> (T,3,3) foot bone frame in camera coords.

    Forward axis heel->big_toe; lateral axis big_toe->small_toe.
    """
    f = mhr70.FOOT_KP[side]
    heel = kp70[:, f["heel"], :]
    big = kp70[:, f["big_toe"], :]
    small = kp70[:, f["small_toe"], :]
    forward = big - heel
    lateral = small - big
    return _frame_from_two(forward, lateral)


def estimate_up_cam(kp70, valid=None):
    """World up (gravity opposite) in camera coords from the mean spine axis."""
    neck = kp70[:, mhr70.MHR70_NAME_TO_IDX["neck"], :]
    lhip = kp70[:, mhr70.MHR70_NAME_TO_IDX["left_hip"], :]
    rhip = kp70[:, mhr70.MHR70_NAME_TO_IDX["right_hip"], :]
    pelvis = 0.5 * (lhip + rhip)
    spine = neck - pelvis
    if valid is not None:
        spine = spine[valid]
    return _normalize(spine.mean(axis=0))


def world_from_cam(up_cam):
    """Rotation mapping camera-frame vectors into a gravity-aligned world (y=up)."""
    y = _normalize(up_cam)
    z_cam = np.array([0.0, 0.0, 1.0])               # camera optical axis
    z = _normalize(z_cam - (z_cam @ y) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=0)               # rows = world axes in cam coords


def _project_so3(C):
    U, _, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def two_sided_procrustes(M, N, iters=15):
    """argmin_{A,B in SO(3)} sum ||M_t - A N_t B||.  M,N: (T,3,3). Returns A,B,deg."""
    B = np.eye(3)
    A = np.eye(3)
    for _ in range(iters):
        A = _project_so3(np.einsum("tij,tkj->ik", M, np.einsum("tjk,kl->tjl", N, B)))
        AN = np.einsum("ij,tjk->tik", A, N)
        B = _project_so3(np.einsum("tji,tjk->ik", AN, M))
    pred = np.einsum("ij,tjk,kl->til", A, N, B)
    rel = np.einsum("tij,tik->tjk", pred, M)        # pred^T M
    tr = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    deg = float(np.degrees(np.arccos(tr)).mean())
    return A, B, deg


def _highpass_gravity(acc, fps, cutoff=0.4):
    """Remove the quasi-constant gravity component via a one-pole high-pass."""
    dt = 1.0 / fps
    rc = 1.0 / (2 * np.pi * cutoff)
    a = rc / (rc + dt)
    out = np.zeros_like(acc)
    prev_in = acc[0]
    out[0] = 0.0
    for i in range(1, len(acc)):
        out[i] = a * (out[i - 1] + acc[i] - prev_in)
        prev_in = acc[i]
    return out


def calibrate_take(take, foot_remove_gravity=None, fit_seconds=None, calib_override=None):
    """Return calibrated world-frame orientation+accel per sensor, plus diagnostics.

    Output dict: ``head_R/lfoot_R/rfoot_R`` (T,3,3) world bone orientation,
    ``head_a/lfoot_a/rfoot_a`` (T,3) world free acceleration (m/s^2), and
    ``calib`` with per-sensor A,B and Procrustes residual (deg).

    ``fit_seconds`` restricts the Procrustes fit to the first N seconds of the
    take (one-time calibration-pose protocol); ``calib_override`` applies a
    pre-fitted per-sensor (A, B) instead of fitting on this take.
    """
    kp = np.asarray(take.gt_joints_mhr70, dtype=np.float64)
    valid = np.asarray(take.gt_valid, dtype=bool) if take.gt_valid is not None \
        else np.ones(len(kp), bool)
    if foot_remove_gravity is None:
        foot_remove_gravity = not take.meta.get("foot_acc_gravity_removed", False)

    Rwc = world_from_cam(estimate_up_cam(kp, valid))
    targets = {
        "head": np.einsum("ij,tjk->tik", Rwc, head_frame_cam(kp)),
        "lfoot": np.einsum("ij,tjk->tik", Rwc, foot_frame_cam(kp, "left")),
        "rfoot": np.einsum("ij,tjk->tik", Rwc, foot_frame_cam(kp, "right")),
    }
    imu_R = {"head": np.asarray(take.imu_head_R, float),
             "lfoot": np.asarray(take.imu_lfoot_R, float),
             "rfoot": np.asarray(take.imu_rfoot_R, float)}
    imu_a = {"head": np.asarray(take.imu_head_acc, float),
             "lfoot": np.asarray(take.imu_lfoot_acc, float),
             "rfoot": np.asarray(take.imu_rfoot_acc, float)}

    fit_idx = valid.copy()
    if fit_seconds is not None:
        n_fit = min(int(round(fit_seconds * float(take.fps))), len(kp))
        fit_idx[n_fit:] = False
    out, calib = {}, {}
    for s in ("head", "lfoot", "rfoot"):
        if calib_override is not None:
            A = np.asarray(calib_override[s]["A"])
            B = np.asarray(calib_override[s]["B"])
            deg = float("nan")
        else:
            A, B, deg = two_sided_procrustes(targets[s][fit_idx], imu_R[s][fit_idx])
        ori = np.einsum("ij,tjk,kl->til", A, imu_R[s], B)
        # world free acceleration: rotate device-frame accel to common world
        aw = np.einsum("ij,tjk,tk->ti", A, imu_R[s], imu_a[s])
        if s != "head" and foot_remove_gravity:
            aw = _highpass_gravity(aw, take.fps)
        out[f"{s}_R"] = ori
        out[f"{s}_a"] = aw
        calib[s] = {"A": A.tolist(), "B": B.tolist(), "resid_deg": deg}
    out["calib"] = calib
    return out
