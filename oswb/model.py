"""Sparse-IMU -> SMPL pose model (IMUPoser-adapted BiLSTM) + shared conventions.

Architecture matches IMUPoser's ``RNN`` (Linear -> 2-layer BiLSTM(512) -> Linear)
so our head(+feet) results are an apples-to-apples adaptation of the published
baseline, with ``n_input = 12 * n_sensors`` and a lower-body output head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Per-sensor feature = 3 (accel) + 9 (orientation rotmat) = 12; sensors head,Lfoot,Rfoot.
N_SENSORS = 3
PER_SENSOR = 12
N_INPUT = N_SENSORS * PER_SENSOR  # 36
ACC_SCALE = 30.0

# Lower-body SMPL joints predicted (matches IMUPoser pred_joints_set["legs"]).
LEGS_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
N_OUT_JOINTS = len(LEGS_JOINTS)
ROT_DIM = 6  # 6D continuous rotation
N_OUTPUT = N_OUT_JOINTS * ROT_DIM  # 54


class PoseRNN(nn.Module):
    def __init__(self, n_input=N_INPUT, n_output=N_OUTPUT, n_hidden=512,
                 n_layers=2, bidirectional=True, dropout=0.2):
        super().__init__()
        self.n_features = n_hidden * (2 if bidirectional else 1)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers,
                           bidirectional=bidirectional, batch_first=True)
        self.linear2 = nn.Linear(self.n_features, n_output)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.linear1(self.dropout(x)))
        x, _ = self.rnn(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.encode(x))


class ContactPoseRNN(PoseRNN):
    """PoseRNN with an auxiliary binary left/right foot-contact head."""

    def __init__(self, *args, n_contacts: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact_head = nn.Linear(self.n_features, n_contacts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.linear2(h), self.contact_head(h)
