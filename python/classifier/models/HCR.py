import numpy as np

from ..dataset import DF
from ..network import Classifier
from ..static import Constant
from .blocks import HCR as HCRBlocks


class HCR(Classifier):
    canJet = ['pt', 'eta', 'phi', 'm']
    notCanJet = ['pt', 'eta', 'phi', 'm', 'isSelJet']
    ancillary = ['nSelJets']
    # ancillary += ['year']
    # ancillary += ['xW', 'xbW']

    def __init__(
            self,
            max_other_jets: int = 8,
    ):
        super().__init__()
        # self.model = HCRBlocks(
        #     device=self.device,
        # )  # TODO

        self.max_other_jets = max_other_jets
        self._df_to_tensor = (
            DF.to_tensor_dataset()
            .add_column(*(
                f'canJet{i}_{k}'
                for k in self.canJet
                for i in range(4)
            ), dtype=np.float32, name='canJet')
            .add_column(*(
                f'notCanJet{i}_{k}'
                for k in self.notCanJet
                for i in range(self.max_other_jets)
            ), dtype=np.float32, name='notCanJet')
            .add_column(*self.ancillary, dtype=np.float32, name='ancillary')
            .add_column(Constant.label_index, dtype=np.uint8)
            .add_column(Constant.weight, dtype=np.float32)
            .add_column(Constant.region_index, dtype=np.uint8)
            .add_column(Constant.event_offset, dtype=np.uint8)
        )

    def dataset(self, data):
        ...  # TODO

    def forward(self, data):
        ...  # TODO

    def loss(self, data):
        ...  # TODO
