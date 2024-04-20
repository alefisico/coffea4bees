from enum import Enum

from classifier.task import Cascade


class InputBranch(Cascade):
    feature_CanJet: list[str] = ["pt", "eta", "phi", "mass"]
    feature_NotCanJet: list[str] = feature_CanJet + ["isSelJet"]
    feature_ancillary: list[str] = ["nSelJets", "year", "xbW", "xW"]
    n_CanJet: int = 4
    n_NotCanJet: int = 8

    @classmethod
    def get__feature_CanJet(cls, var: list[str]):
        return [f"CanJet_{f}" for f in var]

    @classmethod
    def get__feature_NotCanJet(cls, var: list[str]):
        return [f"NotCanJet_{f}" for f in var]


class Input(Cascade):
    label: str = "label"
    region: str = "region"
    weight: str = "weight"
    ancillary: str = "ancillary"
    CanJet: str = "CanJet"
    NotCanJet: str = "NotCanJet"


class Output(Cascade):
    class_score: str = "class_score"
    quadjet_score: str = "quadjet_score"


class MassRegion(Enum):
    SB = 0b10
    ZZSR = 0b0101
    ZHSR = 0b1001
    HHSR = 0b1101
    SR = 0b01


class NTag(Enum):
    fourTag = 0b10
    threeTag = 0b01
