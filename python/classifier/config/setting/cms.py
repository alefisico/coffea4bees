from classifier.task import Cascade


class CollisionData(Cascade):
    eras: dict[str, list[str]] = {
        "UL16_preVFP": ["B", "C", "D", "E", "F"],
        "UL16_postVFP": ["F", "G", "H"],
        "UL17": ["B", "C", "D", "E", "F"],
        "UL18": ["A", "B", "C", "D"],
    }


class TTbarMC(Cascade):
    datasets: list[str] = ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]
