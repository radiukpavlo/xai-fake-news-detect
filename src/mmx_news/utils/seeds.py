from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # sklearn uses numpy RNG by default; if torch is available, set it too
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(deterministic)  # type: ignore[attr-defined]
        torch.backends.cudnn.deterministic = deterministic  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass
