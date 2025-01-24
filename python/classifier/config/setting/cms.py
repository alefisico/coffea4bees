from classifier.task import GlobalSetting


class CollisionData(GlobalSetting):
    "CMS collision data metadata"

    eras: dict[str, list[str]] = {
        "UL16_preVFP": ["B", "C", "D", "E", "F"],
        "UL16_postVFP": ["F", "G", "H"],
        "UL17": ["B", "C", "D", "E", "F"],
        "UL18": ["A", "B", "C", "D"],
    }
    "eras for MC datasets"
    years: list[str] = ["2016", "2017", "2018"]
    "years for data"


class MC_TTbar(GlobalSetting):
    "Metadata for MC sample: TTbar"

    datasets: list[str] = ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]
    "name of TTbar datasets"


class MC_HH_ggF(GlobalSetting):
    "Metadata for MC sample: ggF HH"

    kl: list[float] = [0.0, 1.0, 2.45, 5.0]
