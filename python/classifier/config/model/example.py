from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from classifier.task import ArgParser, parse

from ..scheduler import FinetuneStep, FixedStep
from ..setting.HCR import Input, Output
from ._kfold import KFoldClassifier

if TYPE_CHECKING:
    from classifier.discriminator.HCR import HCRModel
    from torch import Tensor


class _HCR(KFoldClassifier):
    argparser = ArgParser()
    argparser.add_argument(
        "--architecture",
        type=parse.mappings,
        default="",
        help="HCR architecture",
    )
    argparser.add_argument(
        "--ghost-batch",
        type=parse.mappings,
        default="",
        help="ghost batch normalization configuration",
    )
    argparser.add_argument(
        "--training",
        type=parse.mappings,
        default="",
        help="training schedule configuration",
    )
    argparser.add_argument(
        "--finetuning",
        type=parse.mappings,
        default=None,
        help="fine-tuning schedule configuration",
    )

    @staticmethod
    @abstractmethod
    def loss(model: HCRModel, batch: dict[str, Tensor]) -> Tensor:
        pass

    def initializer(self, kfolds: int, offset: int):
        from classifier.discriminator.HCR import GBN, HCRArch, HCRClassifier

        arch = HCRArch(**({"loss": self.loss} | self.opts.architecture))
        gbn = GBN(**self.opts.ghost_batch)
        training = FixedStep(**self.opts.training)
        finetuning = (
            FinetuneStep(**self.opts.finetuning)
            if self.opts.finetuning is not None
            else None
        )

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
        import torch.nn.functional as F

        c_score = batch[Output.class_score]
        weight = batch[Input.weight]
        cross_entropy = F.cross_entropy(c_score, batch[Input.label], reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss
