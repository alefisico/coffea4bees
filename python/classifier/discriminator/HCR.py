# TODO work in progress, match new skimmed data
from torch import Tensor
from torch.utils.data import StackDataset

from ..config.setting.HCR import Architecture, InputBranch
from ..config.state.label import MultiClass
from ..nn.blocks import HCR
from . import Classifier, Module


class HCRModule(Module):
    def __init__(self):
        self._nn = HCR(
            dijetFeatures=Architecture.n_features,
            quadjetFeatures=Architecture.n_features,
            ancillaryFeatures=len(InputBranch.feature_ancillary),
            useOthJets=Architecture.use_attention_block,
            device="cpu",
            nClasses=len(MultiClass.labels),
        )

    @property
    def module(self):
        return self._nn

    def forward(
        self, batch: dict[str, Tensor], validation: bool = False
    ) -> dict[str, Tensor]:
        return super().forward(batch, validation)


class HCRClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self._module: HCRModule = None
        # TODO setMeanStd, setGhostBatches
