from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

from classifier.task import ArgParser, parse

from .._kfold import KFoldClassifier

_SCHEDULER = "classifier.config.scheduler"


if TYPE_CHECKING:
    from classifier.discriminator.roc import MulticlassROC
    from classifier.discriminator.skimmer import BatchType, Splitter
    from torch import Tensor


class HCR(KFoldClassifier):
    loss: Callable[[BatchType], Tensor]
    rocs: Iterable[MulticlassROC] = ()

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
        from classifier.discriminator.HCR import (
            GBNSchedule,
            HCRArch,
            HCRBenchmarks,
            HCRTraining,
        )

        arch = HCRArch(**({"loss": self.loss} | self.opts.architecture))
        gbn = GBNSchedule(**self.opts.ghost_batch)
        training = parse.instance(self.opts.training, _SCHEDULER)
        finetuning = parse.instance(self.opts.finetuning, _SCHEDULER)

        return HCRTraining(
            arch=arch,
            ghost_batch=gbn,
            cross_validation=splitter,
            training_schedule=training,
            finetuning_schedule=finetuning,
            benchmarks=HCRBenchmarks(
                rocs=self.rocs,
            ),
            **kwargs,
        )
