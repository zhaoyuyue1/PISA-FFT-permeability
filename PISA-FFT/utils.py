from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int) -> None:
    """
    Utility to set random seeds for reproducibility as much as possible.

    NOTE:
        This function is provided for convenience, but it is not called by
        default in the training script. Users who require deterministic
        behavior can explicitly pass a seed and invoke this function.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    # We intentionally keep CuDNN benchmark turned on to allow some level
    # of system-dependent behavior.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def medare(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Median Absolute Relative Error (percentage).
    """
    eps = 1e-8
    return float(np.median(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)
