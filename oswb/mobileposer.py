"""MobilePoser-adapted two-stage model (joints -> pose) for head(+feet) IMUs.

Faithful adaptation of MobilePoser's architecture (UIST 2024) to our sensor
layout and lower-body task, so the benchmark has a second, architecture-matched
baseline family next to the IMUPoser-adapted BiLSTM:

  * Stage 1 ``Joints``: RNN(imu 36 -> 9x3 root-relative joints), hidden 256.
  * Stage 2 ``Poser`` : RNN(joints 27 + imu 36 -> 9x6 local rotations), hidden 256.
  * Both stages are 2-layer BiLSTMs with dropout 0.4 (MobilePoser ``RNN``).
  * Training follows their per-module recipe: the poser stage is teacher-forced
    with Gaussian-noised ground-truth joints (std 0.04) while the joints stage
    regresses positions with a temporal smoothness term; at inference the poser
    consumes the joints-stage prediction.

Differences from the upstream repo (deliberate): lower-body 9-joint output
instead of 24, local 6D rotations decoded by the shared SMPL FK (identical
scoring path to the IMUPoser-adapted baseline), and no translation/physics
stages (root translation is out of scope for the lower-body benchmark).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from oswb.model import N_INPUT

N_JOINTS = 9
JOINTS_DIM = N_JOINTS * 3          # 27
POSE_DIM = N_JOINTS * 6            # 54
HIDDEN = 256                       # MobilePoser RNN hidden size
DROPOUT = 0.4                      # MobilePoser RNN dropout
JOINT_NOISE_STD = 0.04             # MobilePoser poser-stage teacher-forcing noise
TEMPORAL_WEIGHT = 1e-5             # MobilePoser t_weight


class MPRNN(nn.Module):
    """MobilePoser's ``RNN``: linear -> ReLU -> dropout -> 2-layer BiLSTM -> linear."""

    def __init__(self, n_input, n_output, n_hidden=HIDDEN, n_layers=2,
                 bidirectional=True, dropout=DROPOUT):
        super().__init__()
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers,
                           bidirectional=bidirectional, batch_first=True)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(torch.relu(self.linear1(x)))
        h, _ = self.rnn(h)
        return self.linear2(h)


class MobilePoserAdapted(nn.Module):
    """Two-stage joints->pose model; ``forward`` returns 6D pose like ``PoseRNN``."""

    def __init__(self, n_input=N_INPUT):
        super().__init__()
        self.joints_rnn = MPRNN(n_input, JOINTS_DIM)
        self.poser_rnn = MPRNN(JOINTS_DIM + n_input, POSE_DIM)

    def predict_joints(self, x):
        return self.joints_rnn(x)

    def forward(self, x, joints=None):
        """``joints`` (B,T,27) enables teacher forcing during training."""
        j = self.joints_rnn(x) if joints is None else joints
        return self.poser_rnn(torch.cat([j, x], dim=-1))


def temporal_loss(seq):
    """MobilePoser's second-difference smoothness penalty (L1 over channels)."""
    acc = seq[:, 2:, :] + seq[:, :-2, :] - 2 * seq[:, 1:-1, :]
    return acc.norm(p=1, dim=2).sum(dim=1).mean()


def jerk_loss(seq):
    """MobilePoser's third-difference jerk penalty on the pose output."""
    jerk = seq[:, 3:, :] - 3 * seq[:, 2:-1, :] + 3 * seq[:, 1:-2, :] - seq[:, :-3, :]
    return jerk.norm(p=1, dim=2).sum(dim=1).mean()
