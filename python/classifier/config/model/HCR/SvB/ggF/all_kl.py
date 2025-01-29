from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.config.setting.cms import MC_HH_ggF
from classifier.config.setting.HCR import Input, Output
from classifier.config.state.label import MultiClass
from classifier.task import ArgParser

from ..._HCR import ROC_BIN, HCREval, HCRTrain, roc_nominal_selection

if TYPE_CHECKING:
    from classifier.ml import BatchType

_BKG = ("multijet", "ttbar")


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

        label = batch[Input.label]
        return torch.isin(label, label.new_tensor(MultiClass.indices(*_BKG, self.sig)))


class _roc_select_ggF(_roc_select_sig):
    def __init__(self, *labels: str, kl: float):
        self.bkg = labels
        self.kl = kl

    def _select(self, batch: BatchType):
        import torch

        label = batch[Input.label]
        return torch.isin(label, label.new_tensor(MultiClass.indices(*self.bkg))) | (
            (batch["kl"] == self.kl) & (label == MultiClass.index("ggF"))
        )


class Train(HCRTrain):
    model = "SvB_ggF-all"
    argparser = ArgParser(description="Train SvB with SM and BSM ggF signals.")

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

    @property
    def rocs(self):
        from classifier.ml.benchmarks.multiclass import ROC

        rocs = [
            ROC(
                name="background vs signal",
                selection=roc_nominal_selection,
                bins=ROC_BIN,
                pos=_BKG,
            )
        ]
        for sig in ("ZZ", "ZH", "ggF"):
            if sig in MultiClass.labels:
                rocs.append(
                    ROC(
                        name=f"background vs {sig}",
                        selection=_roc_select_sig(sig),
                        bins=ROC_BIN,
                        pos=_BKG,
                    )
                )
        if "ggF" in MultiClass.labels:
            for kl in MC_HH_ggF.kl:
                rocs.append(
                    ROC(
                        name=f"background vs ggF (kl={kl:.6g})",
                        selection=_roc_select_ggF(*_BKG, kl=kl),
                        bins=ROC_BIN,
                        pos=_BKG,
                    )
                )
        if "ggF" in MultiClass.trainable_labels:
            for sig in ("ZZ", "ZH"):
                if sig in MultiClass.trainable_labels:
                    for kl in MC_HH_ggF.kl:
                        rocs.append(
                            ROC(
                                name=f"{sig} vs ggF (kl={kl:.6g})",
                                selection=_roc_select_ggF(sig, kl=kl),
                                bins=ROC_BIN,
                                pos=(sig,),
                                neg=("ggF",),
                                score="differ",
                            )
                        )
        if all(sig in MultiClass.trainable_labels for sig in ("ZZ", "ZH")):
            rocs.append(
                ROC(
                    name="ZZ vs ZH",
                    selection=roc_nominal_selection,
                    bins=ROC_BIN,
                    pos=("ZZ",),
                    neg=("ZH",),
                    score="differ",
                )
            )
        return rocs


class Eval(HCREval):
    model = "SvB_ggF-all"

    @staticmethod
    def output_definition(batch: BatchType):
        output = {
            "q_1234": ...,
            "q_1324": ...,
            "q_1423": ...,
            "p_multijet": ...,
            "p_ttbar": ...,
            "p_bkg": batch["p_multijet"] + batch["p_ttbar"],
        }
        for sig in ("ZZ", "ZH", "ggF"):
            sig = f"p_{sig}"
            if sig in batch:
                output[sig] = ...
                if "p_sig" in output:
                    output["p_sig"] = output["p_sig"] + batch[sig]
                else:
                    output["p_sig"] = batch[sig]
        return output
