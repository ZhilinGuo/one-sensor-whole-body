"""Minimal SMPL forward kinematics (torch), matching the IMUPoser convention.

Self-contained re-implementation of the subset of SMPL used by this project:
zero-pose joints/vertices from the official model pickle, forward kinematics
over the 24-joint kinematic tree, and linear blend skinning (no pose
blendshapes, matching IMUPoser's ``ParametricModel`` default). Used both to
synthesize virtual IMU signals from AMASS and to map predicted 6D joint
rotations to 3D joint positions during training and evaluation.

The official SMPL model is license-gated: download ``SMPL v1.0.0`` from
https://smpl.is.tue.mpg.de/ and place ``basicmodel_m_lbs_10_207_0_v1.0.0.pkl``
under ``data/smpl/`` (see INSTALL.md).
"""
from __future__ import annotations

import pickle
import sys
import types

import numpy as np
import torch

_CH_TYPES: list[type] = []
try:  # the official SMPL pickle references chumpy.Ch; use it if installed
    import chumpy

    _CH_TYPES.append(chumpy.Ch)
except ImportError:  # otherwise register a minimal stand-in so unpickling works
    class _Ch(np.ndarray):
        """Stand-in for ``chumpy.Ch`` (an ndarray subclass)."""

        def __new__(cls, *args, **kwargs):
            return np.asarray(args[0] if args else np.zeros(0)).view(cls)

        def __array_finalize__(self, obj):
            pass

        def __setstate__(self, state):
            if isinstance(state, dict):  # chumpy bookkeeping; payload is in .x
                self.__dict__.update(state)
            else:
                super().__setstate__(state)

    _stub = types.ModuleType("chumpy")
    _stub_ch = types.ModuleType("chumpy.ch")
    _stub.Ch = _stub_ch.Ch = _Ch
    _stub.ch = _stub_ch
    sys.modules["chumpy"] = _stub
    sys.modules["chumpy.ch"] = _stub_ch
    _CH_TYPES.append(_Ch)


def _load_smpl_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    def _plain(v):
        if isinstance(v, tuple(_CH_TYPES)):
            x = getattr(v, "x", None)  # chumpy.Ch may keep the payload in .x
            return np.asarray(x) if isinstance(x, np.ndarray) else np.asarray(v)
        return v

    return {k: _plain(v) for k, v in data.items()}


class SMPLModel:
    """SMPL male model (24 joints, 6890 vertices), torch batch ops."""

    def __init__(self, model_file: str, device: torch.device | str = "cpu"):
        data = _load_smpl_pkl(model_file)
        dev = torch.device(device)
        self._J_regressor = torch.from_numpy(np.array(data["J_regressor"].toarray())).float().to(dev)
        self._weights = torch.from_numpy(np.array(data["weights"])).float().to(dev)
        self._shapedirs = torch.from_numpy(np.array(data["shapedirs"])).float().to(dev)
        self._v_template = torch.from_numpy(np.array(data["v_template"])).float().to(dev)
        self._J = torch.from_numpy(np.array(data["J"])).float().to(dev)
        self.faces = np.asarray(data["f"])
        # kintree_table[0] holds parent indices; root's parent is marked -1.
        self.parent = data["kintree_table"][0].tolist()
        self.parent[0] = None
        self.device = dev

    def zero_pose_joints_vertices(self, shape: torch.Tensor | None = None):
        """Zero-pose joints (24,3)/(N,24,3) and vertices, root at the origin."""
        if shape is None:
            j, v = self._J, self._v_template
            return j - j[:1], v - j[:1]
        shape = shape.view(-1, 10).to(self.device)
        v = torch.tensordot(shape, self._shapedirs, dims=([1], [2])) + self._v_template
        j = torch.matmul(self._J_regressor, v)
        return j - j[:, :1], v - j[:, :1]

    def forward_kinematics(self, pose: torch.Tensor, shape: torch.Tensor = None,
                           tran: torch.Tensor = None, calc_mesh: bool = False):
        """SMPL forward kinematics.

        ``pose`` is (N, 24, 3, 3) local rotation matrices, ``shape`` optional
        (N, 10) betas, ``tran`` optional (N, 3) root translation. Returns global
        joint rotations (N, 24, 3, 3) and joint positions (N, 24, 3), plus mesh
        vertices (N, 6890, 3) when ``calc_mesh`` is set.
        """
        pose = pose.view(pose.shape[0], -1, 3, 3).to(self.device)
        n = pose.shape[0]
        j, v = self.zero_pose_joints_vertices(shape)
        # broadcast the rest pose over the batch (shape may be (10,) or (1,10))
        j = j.expand(n, -1, -1) if j.dim() == 2 or j.shape[0] == 1 and n > 1 else j
        v = v.expand(n, -1, -1) if v.dim() == 2 or v.shape[0] == 1 and n > 1 else v

        # local 4x4 transforms: rotation + bone vector (child joint rel. parent)
        bone = j.clone()
        for i in range(1, len(self.parent)):
            bone[:, i] = j[:, i] - j[:, self.parent[i]]
        bottom = torch.zeros(n, pose.shape[1], 1, 4, device=self.device)
        bottom[..., 0, 3] = 1.0
        t_local = torch.cat([torch.cat([pose, bone.unsqueeze(-1)], dim=-1),
                             bottom], dim=-2)                    # (N,J,4,4)

        # global transforms by recursion over the kinematic tree
        t_global = [t_local[:, 0]]
        for i in range(1, len(self.parent)):
            t_global.append(t_global[self.parent[i]] @ t_local[:, i])
        t_global = torch.stack(t_global, dim=1)                  # (N,J,4,4)

        pose_global = t_global[..., :3, :3]
        joint_global = t_global[..., :3, 3]
        if tran is not None:
            joint_global = joint_global + tran.view(-1, 1, 3).to(self.device)
        if not calc_mesh:
            return pose_global, joint_global

        # linear blend skinning (no pose blendshapes)
        rest = torch.cat([j, torch.zeros(n, j.shape[1], 1, device=self.device)], dim=-1)
        t_global = t_global.clone()
        t_global[..., -1:] -= torch.matmul(t_global, rest.unsqueeze(-1))
        t_vertex = torch.tensordot(t_global, self._weights, dims=([1], [1])).permute(0, 3, 1, 2)
        v_h = torch.cat([v, torch.ones(n, v.shape[1], 1, device=self.device)], dim=-1)
        vertex_global = torch.matmul(t_vertex, v_h.unsqueeze(-1)).squeeze(-1)[..., :3]
        if tran is not None:
            vertex_global = vertex_global + tran.view(-1, 1, 3).to(self.device)
        return pose_global, joint_global, vertex_global
