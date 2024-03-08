# TODO work in progress, match new skimmed data
from . import Classifier


class HCR(Classifier):
    def __init__(self):
        super().__init__()
        # TODO

    def forward(self, batch, validation=False): ...  # TODO

    def loss(self, pred): ...
