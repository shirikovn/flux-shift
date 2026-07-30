from __future__ import annotations

import random

import torch


def set_random_seed(seed: int) -> None:
    """
    Initialize random generators used by the experiment.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
