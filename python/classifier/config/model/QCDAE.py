# QCDAE: setup CLI for training and validation
# see classifier.config.model.HCR._HCR for more details (ignore the k-folding part)

# NOTE: Do NOT import large libraries (e.g. torch, numpy, pandas and modules that import them) at the top level of any file in `classifier.config`
# some tasks/functions e.g. help/auto-complete will import this file to get the definition but not run the code, importing large libraries will slow down the process

from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import ArgParser, Model, parse

if TYPE_CHECKING:
    from classifier.ml import BatchType
    from torch import Tensor


class HCR(Model):
    argparser = ArgParser()
    argparser.add_argument(
        "--training",
        nargs="+",
        default=["FixedStep"],
        metavar=("CLASS", "KWARGS"),
        help=f"training scheduler {parse.EMBED}",
    )  # setup the training scheduler from cli
    argparser.add_argument(
        "--other-arg",
    )  # add more arguments as needed

    def train(self):
        from classifier.ml.QCDAE import (
            QCDAETraining,
        )

        training = parse.instance(
            self.opts.training,
            "classifier.config.scheduler",  # where to import the class
        )  # create an instance of training schedule from command input
        ...  # setup more arguments as needed
        return QCDAETraining(
            training_schedule=training,
            arch_args=...,
            loss=self.loss,  # choose a loss function
        )

    @staticmethod
    def loss(batch: BatchType) -> Tensor:
        return ...
