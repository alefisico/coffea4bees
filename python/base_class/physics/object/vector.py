import awkward as ak
import numpy as np

from ...hist import H, Template


class _PlotLorentzVector(Template):
    n       = H((0, 20,             ('n', 'Number')), n=ak.num)
    pt      = H((100, 0, 500,       ('pt', R'$p_{\mathrm{T}}$ [GeV]')))
    mass    = H((100, 0, 500,       ('mass', R'Mass [GeV]')))
    eta     = H((100, -5, 5,        ('eta', R'$\eta$')))
    phi     = H((60, -np.pi, np.pi, ('phi', R'$\phi$')))
    pz      = H((100, -1000, 1000,  ('pz', R'$p_{\mathrm{z}}$ [GeV]')))
    energy  = H((150, 0, 1500,      ('energy', R'Energy [GeV]')))


class _PlotDiLorentzVector(_PlotLorentzVector):
    dr    = H((100, 0, 5, ('dr', R'$\Delta R$')))
    dphi  = H((60, -np.pi, np.pi, ('dphi', R'$\Delta\phi$')))
    st    = H((100, 0, 1000, ('st', R'$S_{\mathrm{T}}$ [GeV]')))


class LorentzVector:
    plot = _PlotLorentzVector
    plot_pair = _PlotDiLorentzVector
