import awkward as ak
from base_class.physics.object import LorentzVector, Jet
from base_class.hist import H, Template


class ClusterHists(Template):
    mA        = H((100, 0, 100, ('mA', "mA [GeV]")))
    mB        = H((100, 0, 100, ('mB', "mB [GeV]")))
    mB_l      = H((100, 0, 300, ('mB', "mB [GeV]")))
    zA        = H((100,  0, 5, ('zA', "z fraction")))
    zA_l      = H((100,  -2, 10, ('zA', "z fraction")))

    decay_phi = H((100, -0.1, 3.2, ('decay_phi', "decay angle")))
    thetaA    = H((100,  0, 2, ('thetaA',    "theta angle")))
    thetaA_l  = H((100,  0, 3.2, ('thetaA',    "theta angle")))    

    n         = H((0, 3,             ('n', 'Number')), n=ak.num)
    
    # mA vs pT_A
    # mB vs pT_B
    # z vs thetaA

    # vs Pt
    
