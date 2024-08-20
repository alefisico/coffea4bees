from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import Cascade

if TYPE_CHECKING:
    from numpy.typing import DTypeLike


class Columns(Cascade):
    event: str = "event"
    weight: str = "weight"
    weight_raw: str = "weight_raw"

    label_index: str = "label_index"
    selection_index: str = "selection_index"

    index_dtype: DTypeLike = "uint8"
