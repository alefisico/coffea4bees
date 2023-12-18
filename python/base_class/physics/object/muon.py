from .lepton import _PlotDiLepton, _PlotLepton


class _PlotCommon:
    ...


class _PlotMuon(_PlotCommon, _PlotLepton):
    ...


class _PlotDiMuon(_PlotCommon, _PlotDiLepton):
    ...


class Muon:
    plot = _PlotMuon
    plot_pair = _PlotDiMuon
