"""Paired per-take significance analysis: head-only vs head+feet.

Joins the per-take rows of two ``cv_*.json`` evaluations on ``(run, seq)`` and
tests whether the rigid-MPJPE difference is systematic rather than fold noise.
Reports Wilcoxon signed-rank and paired t-test p-values (two-sided, Holm
corrected across the four comparisons), mean/median paired differences, and
Cohen's dz. Also renders the 2x2 paired-scatter figure.

Run after the four evaluation pairs (``command/eval_main.sh``):

    python significance.py
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from oswb import config

EVAL_DIR = config.RESULTS_DIR / "eval"
FIG_PATH = config.RESULTS_DIR / "figures" / "paired_head_vs_headfeet.png"
OUT_PATH = EVAL_DIR / "significance_head_vs_headfeet.json"

# (panel title, head-only tag, head+feet tag) — the paper's four comparisons.
COMPARISONS = [
    ("IMUPoser-adapted, leave-one-run-out", "cv_finetune_head_all3", "cv_finetune_all3"),
    ("IMUPoser-adapted, leave-one-motion-out", "cv_finetune_seqout_head_all3", "cv_finetune_seqout_all3"),
    ("MobilePoser-adapted, leave-one-run-out", "cv_mp_head_all3", "cv_mp_all3"),
    ("MobilePoser-adapted, leave-one-motion-out", "cv_mp_seqout_head_all3", "cv_mp_seqout_all3"),
]

MOTION_ORDER = ["gait", "stepping", "turning", "vertical", "composite", "clinical_gait", "upper_limb"]


def _rows_by_take(tag):
    path = EVAL_DIR / f"{tag}.json"
    if not path.exists():
        raise SystemExit(
            f"missing {path} — run command/eval_main.sh first "
            f"(see README: Evaluation)"
        )
    data = json.load(open(path))
    return {(r["run"], r["seq"]): r for r in data["rows"]}


def _holm_adjust(pvals):
    """Holm-Bonferroni adjusted p-values, input order preserved."""
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def analyze_pair(head_tag, feet_tag):
    head_rows = _rows_by_take(head_tag)
    feet_rows = _rows_by_take(feet_tag)
    keys = sorted(set(head_rows) & set(feet_rows))
    head = np.array([head_rows[k]["rigid_mpjpe"] for k in keys])
    feet = np.array([feet_rows[k]["rigid_mpjpe"] for k in keys])
    motions = [head_rows[k]["motion"] for k in keys]
    diff = feet - head  # positive => head-only better
    wil_p = float(stats.wilcoxon(head, feet, alternative="two-sided").pvalue)
    t_p = float(stats.ttest_rel(head, feet).pvalue)
    return {
        "n_takes": len(keys),
        "head_mean": float(head.mean()),
        "feet_mean": float(feet.mean()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "median_diff": float(np.median(diff)),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)),
        "wins_head": int((diff > 0).sum()),
        "wins_feet": int((diff < 0).sum()),
        "wilcoxon_p": wil_p,
        "ttest_p": t_p,
        "takes": {"head": head.tolist(), "feet": feet.tolist(), "motion": motions},
    }


def main():
    results = []
    for title, head_tag, feet_tag in COMPARISONS:
        r = analyze_pair(head_tag, feet_tag)
        r["title"] = title
        r["head_tag"] = head_tag
        r["feet_tag"] = feet_tag
        results.append(r)

    adj = _holm_adjust([r["wilcoxon_p"] for r in results])
    for r, a in zip(results, adj):
        r["wilcoxon_p_holm"] = float(a)
        r["takes_summary_only"] = True
        takes = r.pop("takes")
        r["_scatter"] = takes  # kept for plotting, stripped before save

    # ---- figure: 2x2 paired scatter ----
    cmap = plt.get_cmap("tab10")
    mcolors = {m: cmap(i) for i, m in enumerate(MOTION_ORDER)}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), sharex=False, sharey=False)
    for ax, r in zip(axes.flat, results):
        head = np.array(r["_scatter"]["head"])
        feet = np.array(r["_scatter"]["feet"])
        motions = r["_scatter"]["motion"]
        for m in MOTION_ORDER:
            idx = [i for i, mm in enumerate(motions) if mm == m]
            if idx:
                ax.scatter(head[idx], feet[idx], s=16, alpha=0.85, color=mcolors[m],
                           label=m.replace("_", " "), edgecolors="none")
        lo = min(head.min(), feet.min()) * 0.92
        hi = max(head.max(), feet.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(r["title"], fontsize=9)
        p = r["wilcoxon_p_holm"]
        ptxt = f"$p$={p:.3f}" if p >= 0.001 else f"$p$={p:.1e}"
        ax.text(0.03, 0.97, f"$\\Delta$ med = {r['median_diff']:.1f} mm\n{ptxt} (Wilcoxon, Holm)",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
        ax.set_xlabel("head-only rigid-MPJPE (mm)", fontsize=8)
        ax.set_ylabel("head+feet rigid-MPJPE (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    plt.close(fig)

    for r in results:
        r.pop("_scatter")
    json.dump({"comparisons": results}, open(OUT_PATH, "w"), indent=1)

    for r in results:
        print(f"{r['title']}: n={r['n_takes']} head {r['head_mean']:.1f} vs feet {r['feet_mean']:.1f} mm | "
              f"median diff {r['median_diff']:+.1f} mm, dz={r['cohens_dz']:.2f}, "
              f"wins {r['wins_head']}/{r['wins_head'] + r['wins_feet']}, "
              f"Wilcoxon p={r['wilcoxon_p']:.4f} (Holm {r['wilcoxon_p_holm']:.4f}), t p={r['ttest_p']:.4f}")
    print(f"Saved {OUT_PATH} and {FIG_PATH}")


if __name__ == "__main__":
    main()
