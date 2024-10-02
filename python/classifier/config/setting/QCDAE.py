# QCDAE: add some global settings for QCD autoencoder
# see classifier.config.setting.HCR for more details

from classifier.task import Cascade
from classifier.task.special import WorkInProgress


class InputBranch(WorkInProgress, Cascade):
    feature_Jet: list[str] = ["pt", "eta", "phi", "mass"]
    n_Jet: int = 4

    @classmethod
    def get__feature_Jet(cls, var: list[str]):
        # a workaround for not allowing @classmethod@property anymore after python 3.11
        return [f"Jet_{f}" for f in var]


class Input(WorkInProgress, Cascade):
    # keys used in DataLoader e.g. batch[Input.Jet] == Jet four-momentum tensor
    weight: str = "weight"
    Jet: str = "Jet"  # use variable instead of string makes it easier to track


class Ouput(WorkInProgress, Cascade): ...  # if needed
