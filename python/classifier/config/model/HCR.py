from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from classifier.task import ArgParser, parse

from ..setting.HCR import Input, MassRegion, Output
from ..state.label import MultiClass
from ._kfold import KFoldClassifier

if TYPE_CHECKING:
    from classifier.discriminator.HCR import HCRModel
    from torch import Tensor

_SCHEDULER = "classifier.config.scheduler"


class _HCR(KFoldClassifier):
    argparser = ArgParser()
    argparser.add_argument(
        "--architecture",
        type=parse.mapping,
        default="",
        help="HCR architecture",
    )
    argparser.add_argument(
        "--ghost-batch",
        type=parse.mapping,
        default="",
        help="ghost batch normalization configuration",
    )
    argparser.add_argument(
        "--training",
        nargs="+",
        default=["FixedStep"],
        metavar=("CLASS", "KWARGS"),
        help="training scheduler",
    )
    argparser.add_argument(
        "--finetuning",
        nargs="+",
        default=[],
        metavar=("CLASS", "KWARGS"),
        help="fine-tuning scheduler",
    )

    @staticmethod
    @abstractmethod
    def loss(model: HCRModel, batch: dict[str, Tensor]) -> Tensor:
        pass

    def initializer(self, kfolds: int, offset: int):
        from classifier.discriminator.HCR import GBN, HCRArch, HCRClassifier

        arch = HCRArch(**({"loss": self.loss} | self.opts.architecture))
        gbn = GBN(**self.opts.ghost_batch)
        training = parse.instance(self.opts.training, _SCHEDULER)
        finetuning = parse.instance(self.opts.finetuning, _SCHEDULER)

        return HCRClassifier(
            arch=arch,
            ghost_batch=gbn,
            training_schedule=training,
            finetuning_schedule=finetuning,
            kfolds=kfolds,
            offset=offset,
        )


class FvT(_HCR):
    @staticmethod
    def loss(model: HCRModel, batch: dict[str, Tensor]):
        import torch
        import torch.nn.functional as F

        # get tensors
        c_score = batch[Output.class_score]
        is_SR = (batch[Input.region] & MassRegion.SR.value) != 0
        weight = batch[Input.weight]

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
