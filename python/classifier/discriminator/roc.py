from typing import Callable, Iterable, Literal, TypedDict

import base_class.numpy as npext
from torch import Tensor

from ..algorithm.metrics.roc import FixedThresholdROC, linear_differ
from ..config.state.label import MultiClass
from . import BatchType


class ROCInputType(TypedDict):
    y_true: Tensor
    y_score: Tensor
    weights: Tensor | None


class MulticlassROC(FixedThresholdROC):
    def __init__(
        self,
        name: str,
        selection: Callable[[BatchType], ROCInputType],
        bins: Iterable[float],
        pos: Iterable[str],
        neg: Iterable[str] = None,
        score: Literal["differ"] | None = None,
    ):
        self.name = name
        self.selection = selection
        pos = MultiClass.indices(*pos)
        if neg is not None:
            neg = MultiClass.indices(*neg)
        match score:
            case "differ":
                score = linear_differ
        super().__init__(
            thresholds=bins,
            positive_classes=pos,
            negative_classes=neg,
            score_interpretation=score,
        )

    def update(self, batch: BatchType):
        super().update(**self.selection(batch))

    def roc(self):
        fpr, tpr, auc = super().roc()
        return {
            "name": self.name,
            "FPR": npext.to.base64(fpr),
            "TPR": npext.to.base64(tpr),
            "AUC": auc,
        }
