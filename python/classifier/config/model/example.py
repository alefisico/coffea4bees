from typing import TYPE_CHECKING

from ..scheduler import FixedStep
from ..setting.HCR import Input, Output
from ._kfold import KFoldClassifier

if TYPE_CHECKING:
    from classifier.discriminator.HCR import HCRModule
    from torch import Tensor


def loss_FvT(self: HCRModule, batch: dict[str, Tensor]):
    import torch.nn.functional as F

    c_score = batch[Output.class_score]
    weight = batch[Input.weight]
    cross_entropy = F.cross_entropy(c_score, batch[Input.label], reduction="none")
    loss = (cross_entropy * weight).sum() / weight.sum()
    return loss


class FvT(KFoldClassifier):
    def initializer(self, kfolds: int, offset: int):
        from classifier.discriminator.HCR import HCRClassifier

        return HCRClassifier(
            loss=loss_FvT,
            training_schedule=FixedStep(),
            gbn_size=0,
            kfolds=kfolds,
            offset=offset,
        )
