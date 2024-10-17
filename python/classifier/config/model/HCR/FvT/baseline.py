from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.HCR import Input, MassRegion, Output
from classifier.config.state.label import MultiClass

from .._HCR import ROC_BIN, HCRTrain, roc_nominal_selection

if TYPE_CHECKING:
    from classifier.ml import BatchType


def _roc_data_selection(batch: BatchType):
    import torch

    is_data = batch[Input.label].new(MultiClass.indices("d4", "d3"))
    is_data = torch.isin(batch[Input.label], is_data)
    return {
        "y_pred": batch[Output.class_prob][is_data],
        "y_true": batch[Input.label][is_data],
        "weight": batch[Input.weight][is_data],
    }


class Train(HCRTrain):
    @staticmethod
    def loss(batch: BatchType):
        import torch
        import torch.nn.functional as F

        # get tensors
        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        is_SR = (batch[Input.region] & MassRegion.SR.value) != 0

        # remove 4b data contribution from SR
        no_SR_d4 = torch.ones(
            len(MultiClass.labels), dtype=c_score.dtype, device=c_score.device
        )
        no_SR_d4[MultiClass.labels.index("d4")] = 0

        # calculate loss
        cross_entropy = torch.zeros_like(weight)
        cross_entropy[~is_SR] = F.cross_entropy(
            c_score[~is_SR], batch[Input.label][~is_SR], reduction="none"
        )
        cross_entropy[is_SR] = F.cross_entropy(
            c_score[is_SR], batch[Input.label][is_SR], weight=no_SR_d4, reduction="none"
        )
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from classifier.ml.benchmarks.multiclass import ROC

        return (
            ROC(
                name="4b vs 3b data",
                selection=_roc_data_selection,
                bins=ROC_BIN,
                pos=("d4", "t4"),
            ),
            ROC(
                name="4b vs 3b",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=("d4", "t4"),
            ),
            ROC(
                name="ttbar vs data",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=("t4", "t3"),
            ),
        )
