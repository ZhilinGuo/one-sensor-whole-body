"""Cross-validation engine: alignment helpers, aligned FK fine-tuning loss,
per-take evaluation, and summary statistics. Shared by ``eval.py``,
``mobileposer.py`` and ``eval_contact.py``.
"""
from __future__ import annotations


import collections

import numpy as np
import torch
from torch.utils.data import DataLoader

from oswb import jointset
from oswb import metrics
from oswb import fk
from oswb.dataset import RealWindowDataset


# --------------------------------------------------------------------------- #
# Vectorised alignment helpers (fast eval over long sequences)
# --------------------------------------------------------------------------- #
def pa_mpjpe_fast(pred, gt):
    """Per-frame Procrustes (Umeyama scale+R+t) MPJPE in mm, batched over frames."""
    P, G = np.asarray(pred, float), np.asarray(gt, float)
    muP, muG = P.mean(1, keepdims=True), G.mean(1, keepdims=True)
    P0, G0 = P - muP, G - muG
    cov = np.einsum("tji,tjk->tik", G0, P0) / P.shape[1]
    U, D, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", U, Vt)))
    S = np.repeat(np.eye(3)[None], P.shape[0], 0)
    S[:, 2, 2] = d
    R = np.einsum("tij,tjk,tkl->til", U, S, Vt)
    var = (P0 ** 2).sum((1, 2)) / P.shape[1]
    scale = (D * np.stack([np.ones_like(d), np.ones_like(d), d], 1)).sum(1) / (var + 1e-12)
    aligned = scale[:, None, None] * np.einsum("tij,tkj->tki", R, P0) + muG
    return float(np.linalg.norm(aligned - G, axis=-1).mean() * 1000.0)


def rigid_align(pred, gt):
    """Root-relative pred aligned to gt by a single rotation+scale over the sequence."""
    P = np.asarray(pred, float) - np.asarray(pred, float)[:, :1]
    G = np.asarray(gt, float) - np.asarray(gt, float)[:, :1]
    Pf, Gf = P.reshape(-1, 3), G.reshape(-1, 3)
    cov = Gf.T @ Pf / Pf.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / ((Pf ** 2).sum() / Pf.shape[0] + 1e-12)
    return scale * (P @ R.T), G


def rigid_mpjpe(pred, gt):
    """Root-relative MPJPE (mm) after a single rotation+scale over the sequence."""
    Pa, G = rigid_align(pred, gt)
    return float(np.linalg.norm(Pa - G, axis=-1).mean() * 1000.0)


def rigid_joint_stats(pred, gt, mean_pose):
    """Per-joint rigid error (mm), and SS_res/SS_tot for motion-variance R^2.

    ``mean_pose`` (J,3) is the global root-relative mean pose (the static prior).
    """
    Pa, G = rigid_align(pred, gt)
    err = np.linalg.norm(Pa - G, axis=-1).mean(0) * 1000.0           # (J,)
    prior = np.linalg.norm(mean_pose[None] - G, axis=-1).mean(0) * 1000.0
    ss_res = ((Pa - G) ** 2).sum(-1).sum(0)                          # (J,)
    ss_tot = ((G - G.mean(0)) ** 2).sum(-1).sum(0)
    return err, prior, ss_res, ss_tot, G.shape[0]


# --------------------------------------------------------------------------- #
# Fine-tuning loss (differentiable FK + detached per-window rot+scale alignment)
# --------------------------------------------------------------------------- #
def _solve_rot_scale(Jr, Gr, w):
    """Per-window single R,s aligning root-relative Jr->Gr over valid points (detached)."""
    B = Jr.shape[0]
    Jf, Gf = Jr.reshape(B, -1, 3), Gr.reshape(B, -1, 3)
    wf = w.reshape(B, -1, 1)
    cov = torch.einsum("bni,bnj->bij", Gf * wf, Jf) / (wf.sum(1, keepdim=True) + 1e-6)
    U, D, Vt = torch.linalg.svd(cov)
    d = torch.sign(torch.linalg.det(U @ Vt))
    S = torch.eye(3, device=Jr.device).repeat(B, 1, 1)
    S[:, 2, 2] = d
    R = U @ S @ Vt
    var = ((Jf ** 2) * wf).sum((1, 2)) / (wf.sum((1, 2)) + 1e-6)
    scale = (D * torch.stack([torch.ones_like(d), torch.ones_like(d), d], 1)).sum(1) / (var + 1e-9)
    return R, scale


def aligned_joint_loss(pred6, gtj, mask, device, js, w_override=None):
    """Aligned FK loss over the *supervised* joints of joint set ``js``.

    ``gtj`` is already restricted to the supervised joints (see ``main``), so FK
    outputs are sliced with ``js.sup_idx``; unsupervised spine outputs contribute
    only indirectly, through the FK positions of their supervised descendants.
    On the full set the window alignment is solved on ``js.align_idx()`` (lower
    body, the reliable labels) and the loss is weighted by ``js.loss_weights()``
    (or ``w_override``, e.g. upper-body weights zeroed for staged fine-tuning).
    """
    B, T = pred6.shape[0], pred6.shape[1]
    J = fk.sixd_to_joints(pred6.reshape(B * T, js.n_joints, 6), js.smpl,
                          device=device).reshape(B, T, js.n_joints, 3)
    J = J[:, :, js.sup_idx]
    nj = len(js.sup_idx)
    Jr, Gr = J - J[:, :, :1], gtj - gtj[:, :, :1]
    aidx = js.align_idx() or list(range(nj))
    w = mask[..., None].expand(-1, -1, nj)[:, :, aidx]   # alignment weights
    R, s = _solve_rot_scale(Jr.detach()[:, :, aidx], Gr[:, :, aidx], w)
    Ja = s[:, None, None, None] * torch.einsum("bij,btkj->btki", R, Jr)
    err = torch.linalg.norm(Ja - Gr, dim=-1)            # (B,T,nj)
    lw = w_override if w_override is not None else js.loss_weights()
    lw_t = torch.tensor(lw, device=err.device) if lw else torch.ones(nj, device=err.device)
    m = mask[..., None] * lw_t
    return (err * m).sum() / (m.sum() + 1e-6)


# --------------------------------------------------------------------------- #
def evaluate(model, takes, device, mean_pose, js, group_pos=None, eval_pos=None):
    """Per-take metrics. ``rigid_*``/``ss_*`` use the alignment solved over all
    evaluated joints. For joint groups (lower/upper on the full set) we also
    store matching-basis stats: the similarity transform is re-solved on the
    group's joints alone, so group errors are comparable across models with
    different output sets. ``eval_pos`` selects the evaluated subset within the
    sup-sliced arrays (loss-only spine slots, when supervised, are excluded).
    """
    model.eval()
    rows = []
    for ta in takes:
        X = torch.as_tensor(ta.X, dtype=torch.float32)[None].to(device)
        with torch.no_grad():
            pred = model(X)[0].cpu()
        J_full = fk.sixd_to_joints(pred.reshape(-1, js.n_joints, 6), js.smpl).numpy()
        J = J_full[:, js.sup_idx]
        if eval_pos is not None:
            J = J[:, eval_pos]
        v = ta.valid
        G = ta.Yj[:, eval_pos] if eval_pos is not None else ta.Yj
        Jv, Gv = J[v], G[v]
        rows.append(_score_take(ta, Jv, Gv, J, mean_pose, group_pos))
    return rows


def _score_take(ta, Jv, Gv, J, mean_pose, group_pos):
    err, prior, ss_res, ss_tot, n = rigid_joint_stats(Jv, Gv, mean_pose)
    row = {
        "run": ta.run, "seq": ta.seq, "motion": ta.motion_class,
        "pa_mpjpe": pa_mpjpe_fast(Jv, Gv),
        "rigid_mpjpe": rigid_mpjpe(Jv, Gv),
        "mpjve": metrics.mpjve(Jv, Gv, ta.fps, "root"),
        "jitter": metrics.jitter(J, ta.fps),
        "joint_err": err.tolist(), "prior_err": prior.tolist(),
        "ss_res": ss_res.tolist(), "ss_tot": ss_tot.tolist(),
        "n": int(len(Jv)),
    }
    for gname, gidx in (group_pos or {}).items():
        Js, Gs = Jv[:, gidx], Gv[:, gidx]
        ms = mean_pose[gidx] - mean_pose[gidx[0]]
        ge, gp, gsr, gst, _ = rigid_joint_stats(Js, Gs, ms)
        row[f"{gname}_rigid_mpjpe"] = rigid_mpjpe(Js, Gs)
        row[f"{gname}_joint_err"] = ge.tolist()
        row[f"{gname}_prior_err"] = gp.tolist()
        row[f"{gname}_ss_res"] = gsr.tolist()
        row[f"{gname}_ss_tot"] = gst.tolist()
    return row


def evaluate_prior(takes, mean_pose, js, group_pos=None, eval_pos=None):
    """Constant mean-pose baseline: predict the global root-relative mean pose.

    Scored root-relative with no alignment solved (a constant prediction makes
    the rigid/Procrustes alignment degenerate); PA-MPJPE, MPJVE and jitter are
    therefore not reported (Table 1 dashes).
    """
    rows = []
    for ta in takes:
        v = ta.valid
        G = ta.Yj[:, eval_pos] if eval_pos is not None else ta.Yj
        Gv = G[v] - G[v][:, :1]                       # root-relative GT
        nj = Gv.shape[1]
        err = np.linalg.norm(mean_pose[None] - Gv, axis=-1).mean(0) * 1000.0
        ss_res = ((mean_pose[None] - Gv) ** 2).sum(-1).sum(0)
        ss_tot = ((Gv - Gv.mean(0)) ** 2).sum(-1).sum(0)
        row = {
            "run": ta.run, "seq": ta.seq, "motion": ta.motion_class,
            "pa_mpjpe": float("nan"),
            "rigid_mpjpe": float(err.mean()),
            "mpjve": float("nan"),
            "jitter": float("nan"),
            "joint_err": err.tolist(), "prior_err": err.tolist(),
            "ss_res": ss_res.tolist(), "ss_tot": ss_tot.tolist(),
            "n": int(v.sum()),
        }
        for gname, gidx in (group_pos or {}).items():
            Gs = Gv[:, gidx]
            ms = mean_pose[gidx] - mean_pose[gidx[0]]
            ge = np.linalg.norm(ms[None] - Gs, axis=-1).mean(0) * 1000.0
            row[f"{gname}_rigid_mpjpe"] = float(ge.mean())
            row[f"{gname}_joint_err"] = ge.tolist()
            row[f"{gname}_prior_err"] = ge.tolist()
            row[f"{gname}_ss_res"] = ((ms[None] - Gs) ** 2).sum(-1).sum(0).tolist()
            row[f"{gname}_ss_tot"] = ((Gs - Gs.mean(0)) ** 2).sum(-1).sum(0).tolist()
        rows.append(row)
    return rows


def global_mean_pose(takes):
    allp = np.concatenate([ta.Yj[ta.valid] - ta.Yj[ta.valid][:, :1] for ta in takes])
    return allp.mean(0)


def _inject_foot_bias(takes, sigma_deg, seed=20260730):
    """Constant per-take foot mounting bias (body-frame), i.i.d. across takes/feet.

    Models session-to-session sensor-to-bone calibration spread: each take gets
    a fixed random rotation per foot with per-axis std ``sigma_deg`` degrees,
    applied to the orientation channels of the packed 36-D input.
    """
    from oswb import rotation as rot
    rng = np.random.default_rng(seed)
    sigma = np.deg2rad(sigma_deg)
    for ta in takes:
        X = ta.X.astype(np.float64)
        for s in (1, 2):  # ori blocks: lfoot cols 18:27, rfoot cols 27:36
            blk = X[:, 9 + s * 9:9 + (s + 1) * 9].reshape(-1, 3, 3)
            omega = rng.normal(0.0, sigma, 3)
            dR = rot.axis_angle_to_matrix_torch(
                torch.from_numpy(omega)[None]).numpy()[0]
            blk[:] = blk @ dR
        ta.X = X.astype(np.float32)


def finetune(model, train_takes, device, epochs, lr, batch, win, js,
             stage_lower=0, freeze_stage2=False):
    """Per-fold fine-tuning. With ``stage_lower > 0``, the first ``stage_lower``
    epochs train with upper-body loss weights zeroed (the proven lower-body
    recipe), then the remaining epochs use the full weighted loss; with
    ``freeze_stage2`` the encoder (linear1+rnn) is frozen for stage 2 so only
    the output head adapts to the widened task.
    """
    ds = RealWindowDataset(train_takes, win=win)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4,
                    drop_last=len(ds) > batch)

    def run_epochs(n_epochs, w_override):
        model.train()
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)
        for _ in range(n_epochs):
            for X, Yj, M in dl:
                X, Yj, M = X.to(device), Yj.to(device), M.to(device)
                opt.zero_grad()
                loss = aligned_joint_loss(model(X), Yj, M, device, js, w_override)
                loss.backward()
                opt.step()

    lw = js.loss_weights()
    if stage_lower <= 0 or lw is None:
        run_epochs(epochs, None)
        return model

    upper = set(jointset.FULL_UPPER_IDX)
    w_stage1 = [0.0 if out_i in upper else w for w, out_i in zip(lw, js.sup_idx)]
    run_epochs(stage_lower, w_stage1)
    if freeze_stage2:
        for name, p in model.named_parameters():
            if not name.startswith("linear2"):
                p.requires_grad_(False)
    run_epochs(epochs - stage_lower, None)
    return model


def summarize(rows, tag, joint_names, groups=None):
    nj = len(joint_names)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    ssr = collections.defaultdict(lambda: np.zeros(nj)); sst = collections.defaultdict(lambda: np.zeros(nj))
    for r in rows:
        for k in ("pa_mpjpe", "rigid_mpjpe", "mpjve", "jitter"):
            by[r["motion"]][k].append(r[k]); by["ALL"][k].append(r[k])
        by[r["motion"]]["joint_err"].append(r["joint_err"]); by["ALL"]["joint_err"].append(r["joint_err"])
        by[r["motion"]]["prior_err"].append(r["prior_err"]); by["ALL"]["prior_err"].append(r["prior_err"])
        for mc in (r["motion"], "ALL"):
            ssr[mc] += np.array(r["ss_res"]); sst[mc] += np.array(r["ss_tot"])
    print(f"\n=== {tag}: rigid-MPJPE / R^2(motion) / prior-MPJPE / PA (by motion) ===")
    out = {}
    for mc in sorted(by):
        rg = np.mean(by[mc]["rigid_mpjpe"]); pa = np.mean(by[mc]["pa_mpjpe"])
        vv = np.nanmean(by[mc]["mpjve"]); n = len(by[mc]["pa_mpjpe"])
        prior = np.mean([np.mean(x) for x in by[mc]["prior_err"]])
        r2 = float((1 - ssr[mc] / (sst[mc] + 1e-12)).mean())
        print(f"  {mc:16s} rigid={rg:6.1f} R2={r2:5.2f} prior={prior:6.1f} PA={pa:6.1f} MPJVE={vv:7.1f} (n={n})")
        out[mc] = {"rigid_mpjpe": rg, "r2_motion": r2, "prior_mpjpe": prior,
                   "pa_mpjpe": pa, "mpjve": vv, "n": n}
    # per-joint for ALL
    je = np.mean(np.array(by["ALL"]["joint_err"]), 0)
    pe = np.mean(np.array(by["ALL"]["prior_err"]), 0)
    r2j = 1 - ssr["ALL"] / (sst["ALL"] + 1e-12)
    print("  per-joint (rigid mm | prior mm | R2):")
    for i, jn in enumerate(joint_names):
        print(f"     {jn:14s} {je[i]:6.1f} | {pe[i]:6.1f} | {r2j[i]:5.2f}")
    out["per_joint"] = {"names": joint_names, "rigid_mm": je.tolist(),
                        "prior_mm": pe.tolist(), "r2": r2j.tolist()}
    if groups:
        # Matching-basis group aggregates (alignment re-solved per group), with
        # pooled R^2 across takes.
        gout = {}
        for gname, gidx in groups.items():
            key = f"{gname}_rigid_mpjpe"
            gr = float(np.mean([r[key] for r in rows]))
            gprior = float(np.mean([np.mean(r[f"{gname}_prior_err"]) for r in rows]))
            gsr = sum(np.array(r[f"{gname}_ss_res"]) for r in rows)
            gst = sum(np.array(r[f"{gname}_ss_tot"]) for r in rows)
            gr2 = float((1 - gsr / (gst + 1e-12)).mean())
            gout[gname] = {"rigid_mpjpe": gr, "prior_mpjpe": gprior, "r2_motion": gr2}
            print(f"  group {gname:6s} rigid={gr:6.1f} prior={gprior:6.1f} R2={gr2:5.2f} (matching basis)")
        out["groups"] = gout
    return out


