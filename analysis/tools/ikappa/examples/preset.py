StackGroups = [
    (
        "Background",
        [
            "TTTo2L2Nu",
            "TTToSemiLeptonic",
            "TTToHadronic",
            "QCD Multijet",
        ],
    ),
]
ModelPatterns = {
    "ggF": [
        r"^GluGluToHHTo4B_cHHH(?P<kl>[p\d]+)$",
    ],
}
GlyphVisibility = [
    ("data", "errorbar"),
    ("QCD Multijet", "fill"),
    ("TTTo2L2Nu", "fill"),
    ("TTToSemiLeptonic", "fill"),
    ("TTToHadronic", "fill"),
    ("ZH4b", "step"),
    ("ZZ4b", "step"),
    (None, "step"),
]
SelectedHists = [
    r"SvB_MA\.ps.*fine",
]
SelectedCategories = {
    "failSvB": [True, False],
    "passSvB": [True, False],
    "region": [0, 1],
    "tag": [4],
    "year": ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18", "2016", "2017", "2018"],
}
