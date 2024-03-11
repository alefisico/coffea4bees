from classifier.task import Cascade


class InputBranch(Cascade):
    feature_CanJet: list[str] = ["pt", "eta", "phi", "mass"]
    feature_NotCanJet: list[str] = feature_CanJet + ["isSelJet"]
    feature_ancillary: list[str] = ["nSelJets", "year"]
    n_CanJet: int = 4
    n_NotCanJet: int = 8

    @classmethod
    def _feature_CanJet(cls, var: list[str]):
        return [f"CanJet_{f}" for f in var]

    @classmethod
    def _feature_NotCanJet(cls, var: list[str]):
        return [f"NotCanJet_{f}" for f in var]


class Architecture(Cascade):
    n_features: int = 8
    use_attention_block: bool = True
