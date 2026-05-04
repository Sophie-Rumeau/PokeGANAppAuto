from __future__ import annotations

import random
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
