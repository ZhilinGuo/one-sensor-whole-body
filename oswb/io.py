"""Small IO helpers for the canonical per-take ``.npz`` tensors."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz(path: str | Path, **arrays: Any) -> Path:
    """Save a compressed ``.npz``; dict-valued entries are JSON-stringified."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for k, v in arrays.items():
        if isinstance(v, dict):
            payload[k] = np.array(json.dumps(v))
        else:
            payload[k] = v
    np.savez_compressed(path, **payload)
    return path


def load_npz(path: str | Path) -> dict:
    """Load an ``.npz`` written by :func:`save_npz`, restoring JSON dicts."""
    import json

    data = dict(np.load(path, allow_pickle=True))
    out: dict = {}
    for k, v in data.items():
        if v.dtype.kind == "U" or (v.ndim == 0 and v.dtype == object):
            try:
                out[k] = json.loads(str(v))
                continue
            except (ValueError, TypeError):
                pass
        out[k] = v
    return out


def take_id(run: int, seq: int) -> str:
    """Canonical take identifier, e.g. ``run3_seq5``."""
    return f"run{run}_seq{seq}"
