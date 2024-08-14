from __future__ import annotations

from typing import TYPE_CHECKING

from ...setting.cms import MC_HH_ggF
from ...setting.HCR import Input, Output
from ...state.label import MultiClass
from ._HCR import HCR
from .FvT import _ROC_BIN, _roc_nominal_selection

if TYPE_CHECKING:
    from classifier.ml.skimmer import BatchType

_BKG = ("data", "ttbar")


class _roc_select_sig:
    def __init__(self, sig: str):
        self.sig = sig

    def __call__(self, batch: BatchType):
        selected = self._select(batch)
        return {
            "y_pred": batch[Output.class_prob][selected],
            "y_true": batch[Input.label][selected],
            "weight": batch[Input.weight][selected],
        }

    def _select(self, batch: BatchType):
        import torch

        return torch.isin(batch[Input.label], MultiClass.indices(*_BKG, self.sig))


class _roc_select_ggF(_roc_select_sig):
    def __init__(self, *labels: str, kl: float):
        self.bkg = labels
        self.kl = kl

    def _select(self, batch: BatchType):
        import torch

        return torch.isin(batch[Input.label], MultiClass.indices(*self.bkg)) | (
            (batch["kl"] == self.kl) & (batch[Input.label] == MultiClass.index("ggF"))
        )


class ggF_SM(HCR):
    @staticmethod
    def loss(batch: BatchType):
        import torch.nn.functional as F

        bkg_sigsm = ~(
            (batch[Input.label] == MultiClass.index("ggF")) & (batch["kl"] != 1.0)
        )

        c_score = batch[Output.class_raw][bkg_sigsm]
        weight = batch[Input.weight][bkg_sigsm]
        label = batch[Input.label][bkg_sigsm]

        # calculate loss
        cross_entropy = F.cross_entropy(c_score, label, reduction="none")
        loss = (cross_entropy * weight).sum() / weight.sum()
        return loss

    @property
    def rocs(self):
        from classifier.ml.benchmarks.multiclass import ROC

        return [
            ROC(
                name="background vs signal",
                selection=_roc_nominal_selection,
                bins=_ROC_BIN,
                pos=_BKG,
            ),
            *(
                ROC(
                    name=f"background vs {sig}",
                    selection=_roc_select_sig(sig),
                    bins=_ROC_BIN,
                    pos=_BKG,
                )
                for sig in ("ZZ", "ZH", "ggF")
            ),
            *(
                ROC(
                    name=f"background vs ggF (kl={kl:.6g})",
                    selection=_roc_select_ggF(*_BKG, kl=kl),
                    bins=_ROC_BIN,
                    pos=_BKG,
                )
                for kl in MC_HH_ggF.kl
            ),
            *(
                ROC(
                    name=f"{sig} vs ggF (kl={kl:.6g})",
                    selection=_roc_select_ggF(sig, kl=kl),
                    bins=_ROC_BIN,
                    pos=(sig,),
                    neg=("ggF",),
                    score="differ",
                )
                for sig in ("ZZ", "ZH")
                for kl in MC_HH_ggF.kl
            ),
            ROC(
                name="ZZ vs ZH",
                selection=_roc_nominal_selection,
                bins=_ROC_BIN,
                pos=("ZZ",),
                neg=("ZH",),
                score="differ",
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
