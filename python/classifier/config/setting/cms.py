from classifier.task import Cascade


class CollisionData(Cascade):
    eras: dict[str, list[str]] = {
        "UL16_preVFP": ["B", "C", "D", "E", "F"],
        "UL16_postVFP": ["F", "G", "H"],
        "UL17": ["B", "C", "D", "E", "F"],
        "UL18": ["A", "B", "C", "D"],
    }
    years: list[str] = ["2016", "2017", "2018"]


class MC_TTbar(Cascade):
    datasets: list[str] = ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]


class MC_HH_ggF(Cascade):
    kl: list[float] = [0.0, 1.0, 2.45, 5.0]
