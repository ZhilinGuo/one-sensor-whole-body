"""Filesystem paths and canonical joint/sensor definitions.

Layout (created on demand; ``data/`` and ``results/`` are gitignored)::

    <repo>/data/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl   (download, INSTALL.md)
    <repo>/data/amass/<Dataset>/**_poses.npz                (download, INSTALL.md)
    <repo>/data/processed/run{R}_seq{S}.npz                 (benchmark release)
    <repo>/data/synthetic/AMASS/<Dataset>/{pose,shape,tran,joint,vrot,vacc}.pt
    <repo>/results/baselines/<name>/ckpt.pt                 (pretraining output)
    <repo>/results/eval/cv_<tag>.json                       (evaluation output)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("OSWB_DATA_DIR", REPO_ROOT / "data"))
PROCESSED_DIR = DATA_DIR / "processed"          # benchmark per-take .npz
SYNTH_DIR = DATA_DIR / "synthetic"              # synthesized AMASS IMU tensors
AMASS_DIR = DATA_DIR / "amass"                  # raw AMASS download
SMPL_MALE_PKL = DATA_DIR / "smpl" / "basicmodel_m_lbs_10_207_0_v1.0.0.pkl"

RESULTS_DIR = Path(os.environ.get("OSWB_RESULTS_DIR", REPO_ROOT / "results"))

# AMASS subsets used for synthetic pretraining (paper Section 4).
AMASS_DATASETS = ["CMU", "BioMotionLab_NTroje", "MPI_HDM05"]

# --------------------------------------------------------------------------- #
# SMPL skeleton (24 body joints)
# --------------------------------------------------------------------------- #
SMPL_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# SMPL mesh vertex ids for synthetic-IMU acceleration placement.
# head 411 (as DIP/IMUPoser); feet use foot-segment vertices (smpl_vert_segmentation
# leftFoot[0]=3327, rightFoot[0]=6727).
VERT_HEAD = 411
VERT_LEFT_FOOT = 3327
VERT_RIGHT_FOOT = 6727
# SMPL joints for synthetic-IMU orientation: head=15, foot segment oriented by the
# ankle joints (7=left_ankle, 8=right_ankle), matching an in-shoe insole.
JOINT_HEAD = 15
JOINT_LEFT_FOOT = 7
JOINT_RIGHT_FOOT = 8
# Worn-sensor order used everywhere downstream: head, left foot, right foot.
SENSOR_ORDER = ["head", "left_foot", "right_foot"]

# Motion protocol grouping for per-class evaluation.
SEQ_MOTION_CLASS = {
    1: "gait", 2: "stepping", 3: "turning", 4: "vertical",
    5: "composite", 6: "clinical_gait", 7: "upper_limb",
}


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, SYNTH_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
