from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.HCR import Input
from classifier.config.setting.ml import SplitterKeys
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser

from . import all_kl

if TYPE_CHECKING:
    from classifier.ml import BatchType


def _remove_non_sm_ggF(batch: BatchType):
    return ~((batch[Input.label] == MultiClass.index("ggF")) & (batch["kl"] != 1.0))


class Train(all_kl.Train):
    model = "SvB_ggF_baseline"
    argparser = ArgParser(description="Train SvB with SM ggF signal.")

    def initializer(self, splitter, **kwargs):
        from classifier.ml.skimmer import Filter

        return super().initializer(
            splitter=Filter(**{SplitterKeys.training: _remove_non_sm_ggF}) + splitter,
            **kwargs,
        )


class Eval(all_kl.Eval):
    model = "SvB_ggF_baseline"
