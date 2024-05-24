from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import numpy.typing as npt


def to_num(tensor: torch.Tensor):
    return tensor.detach().cpu().item()


def to_arr(tensor: torch.Tensor) -> npt.NDArray:
    return tensor.detach().cpu().numpy()
