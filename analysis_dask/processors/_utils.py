from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def selection_to_label(
    selections: dict[str, npt.NDArray],
    size: int,
    default: str = "other",
) -> npt.NDArray:
    """
    A simple function to map boolean selections to string array.

    **This function assumes the following**:
      - The selections are exclusive.
      - The size matches all boolean selection arrays.
    """
    labels = list(selections)  # guarantee order
    label_val = np.array([default, *labels])
    label_idx = np.zeros(size, dtype=int)
    for i, label in enumerate(labels):
        label_idx[selections[label]] = i + 1
    return label_val[label_idx]
