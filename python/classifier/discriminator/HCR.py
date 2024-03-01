# TODO work in progress, match new skimmed data
from . import Classifier


class HCR(Classifier):
    def __init__(
        self,
        max_other_jets: int = 8,
    ):
        super().__init__()
        # TODO

    def forward(self, data):
        ...  # TODO

    def loss(self, data):
        ...  # TODO
