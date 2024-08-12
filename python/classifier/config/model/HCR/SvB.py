from __future__ import annotations

from typing import TYPE_CHECKING

from ...setting.cms import MC_HH_ggF
from ...setting.HCR import Input, Output
from ...state.label import MultiClass
from ._HCR import HCR
from .FvT import _ROC_BIN, _roc_nominal_selection

if TYPE_CHECKING:
    from classifier.ml.skimmer import BatchType


class _roc_select_kl:
    def __init__(self, signal: str, kl: float):
        self.signal = signal
        self.kl = kl

    def __call__(self, batch: BatchType):
        selected = self._select(batch)
        return {
            "y_pred": batch[Output.class_prob][selected],
            "y_true": batch[Input.label][selected],
            "weight": batch[Input.weight][selected],
        }

    def _select(self, batch: BatchType):
        is_signal = batch[Input.label] == MultiClass.index(self.signal)
        not_kl = batch["kl"] != self.kl

        return ~(is_signal & not_kl)


class ggF_SM(HCR):
    @staticmethod
    def loss(batch: BatchType):
        import torch.nn.functional as F

        bkg_sigsm = _roc_select_kl("ggF", 1.0)._select(batch)

        c_score = batch[Output.class_raw][bkg_sigsm]
        weight = batch[Input.weight][bkg_sigsm]
        label = batch[Input.label][bkg_sigsm]

        # calculate loss
        cross_entropy = F.cross_entropy(c_score, label, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from classifier.ml.roc import MulticlassROC

        return [
            MulticlassROC(
                name="background vs signal",
                selection=_roc_nominal_selection,
                bins=_ROC_BIN,
                pos=("data", "ttbar"),
            ),
            *(
                MulticlassROC(
                    name=f"background vs {sig}",
                    selection=_roc_nominal_selection,
                    bins=_ROC_BIN,
                    pos=(sig,),
                    neg=("data", "ttbar"),
                    score="differ",
                )
                for sig in ("ZZ", "ZH", "ggF")
            ),
            *(
                MulticlassROC(
                    name=f"background vs ggF (kl={kl:.6g})",
                    selection=_roc_select_kl("ggF", kl),
                    bins=_ROC_BIN,
                    pos=("ggF",),
                    neg=("data", "ttbar"),
                    score="differ",
                )
                for kl in MC_HH_ggF.kl
            ),
        ]


class ggF_All(ggF_SM):
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
