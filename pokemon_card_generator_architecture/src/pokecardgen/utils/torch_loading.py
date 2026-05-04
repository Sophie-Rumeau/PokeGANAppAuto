from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def torch_load_any(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """
    Charge les anciens checkpoints torch.save(obj, ...) même avec PyTorch récent.
    PyTorch >=2.6 peut imposer weights_only=True par défaut selon le contexte.
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
