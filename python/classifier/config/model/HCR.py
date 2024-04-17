from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from classifier.task import ArgParser, parse

from ..setting.HCR import Input, MassRegion, Output
from ..state.label import MultiClass
from ._kfold import KFoldClassifier

if TYPE_CHECKING:
    from classifier.discriminator.skimmer import Splitter
    from torch import Tensor

_SCHEDULER = "classifier.config.scheduler"


class _HCR(KFoldClassifier):
    loss: Callable[[dict[str, Tensor]], Tensor]

    argparser = ArgParser()
    argparser.add_argument(
        "--architecture",
        type=parse.mapping,
        default="",
        help=f"HCR architecture {parse.EMBED}",
    )
    argparser.add_argument(
        "--ghost-batch",
        type=parse.mapping,
        default="",
        help=f"ghost batch normalization configuration {parse.EMBED}",
    )
    argparser.add_argument(
        "--training",
        nargs="+",
        default=["FixedStep"],
        metavar=("CLASS", "KWARGS"),
        help=f"training scheduler {parse.EMBED}",
    )
    argparser.add_argument(
        "--finetuning",
        nargs="+",
        default=[],
        metavar=("CLASS", "KWARGS"),
        help=f"fine-tuning scheduler {parse.EMBED}",
    )

    def initializer(self, splitter: Splitter, **kwargs):
        from classifier.discriminator.HCR import GBN, HCRArch, HCRClassifier

        arch = HCRArch(**({"loss": self.loss} | self.opts.architecture))
        gbn = GBN(**self.opts.ghost_batch)
        training = parse.instance(self.opts.training, _SCHEDULER)
        finetuning = parse.instance(self.opts.finetuning, _SCHEDULER)

        return HCRClassifier(
            arch=arch,
            ghost_batch=gbn,
            cross_validation=splitter,
            training_schedule=training,
            finetuning_schedule=finetuning,
            **kwargs,
        )


class FvT(_HCR):
    @staticmethod
    def loss(batch: dict[str, Tensor]):
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
