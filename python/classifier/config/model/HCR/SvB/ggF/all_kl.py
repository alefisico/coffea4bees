from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.HCR import Input, Output
from classifier.task import ArgParser

from . import baseline

if TYPE_CHECKING:
    from classifier.ml import BatchType


class Train(baseline.Train):
    argparser = ArgParser(description="Train SvB with SM and BSM ggF signals.")
    model = "SvB_ggF_all_kl"

    @staticmethod
    def loss(batch: BatchType):
        import torch.nn.functional as F

        c_score = batch[Output.class_raw]
        weight = batch[Input.weight]
        label = batch[Input.label]

        # calculate loss
        cross_entropy = F.cross_entropy(c_score, label, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss


class Eval(baseline.Eval):
    model = "SvB_ggF_all_kl"
