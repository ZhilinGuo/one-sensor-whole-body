"""MHR70 keypoint layout (SAM 3D Body pseudo-GT convention).

SAM 3D Body's ``pred_keypoints_3d`` first 70 keypoints, in camera coordinates,
metres. Only the name->index map and foot-keypoint indices are needed here:
calibration builds bone frames from them, and the full-body joint set maps
them to SMPL-style targets.
"""
from __future__ import annotations

MHR70_NAME_TO_IDX = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14,
    "left_big_toe": 15, "left_small_toe": 16, "left_heel": 17,
    "right_big_toe": 18, "right_small_toe": 19, "right_heel": 20,
    "left_wrist": 62, "right_wrist": 41, "neck": 69,
}

FOOT_KP = {
    "left": {"big_toe": 15, "small_toe": 16, "heel": 17, "ankle": 13},
    "right": {"big_toe": 18, "small_toe": 19, "heel": 20, "ankle": 14},
}
