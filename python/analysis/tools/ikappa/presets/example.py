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
VisibleGlyphs = [
    ("data", "errorbar"),
    ("QCD Multijet", "fill"),
    ("TTTo2L2Nu", "fill"),
    ("TTToSemiLeptonic", "fill"),
    ("TTToHadronic", "fill"),
    ("ZH4b", "step"),
    ("ZZ4b", "step"),
]
SelectedHists = [
    r"SvB_MA\.ps.*fine",
]
