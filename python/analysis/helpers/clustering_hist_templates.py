import awkward as ak
from base_class.physics.object import LorentzVector, Jet
from base_class.hist import H, Template


class ClusterHists(Template):
    pt        = H((100,  0, 300, ('pt',    "pt [GeV]")))
    pt_l      = H((100,  0, 500, ('pt',    "pt [GeV]")))

    pz        = H((100, -500, 500, ('pz',    "pz [GeV]")))


    eta        = H((50,  -3, 3, ('eta',    "eta")))

    mA        = H((100, 0, 50, ('part_A.mass', "mA [GeV]")))
    rhoA      = H((100, 0, 0.5,  ('rhoA', "rho A (mass/pt)")))

    mB        = H((100, 0, 50, ('mB', "mB [GeV]")))
    rhoB      = H((100, 0, 1,  ('rhoB', "rho B (mass/pt)")))
    mB_l      = H((100, 0, 300, ('mB', "mB [GeV]")))
    mA_l      = H((100, 0, 300, ('mA', "mA [GeV]")))

    mA_vs_mB   = H((50, 0, 50, ('part_A.mass', 'Mass A [GeV]')),
                   (50, 0, 50, ('part_B.mass', 'Mass A [GeV]')))

    mA_vs_pTA   = H((50, 0, 50,  ('part_A.mass', 'Mass A [GeV]')),
                    (50, 0, 250, ('part_A.pt', '$p_T$ A [GeV]')))

    rhoA_vs_pTA   = H((50, 0, 1,  ('rhoA', 'rho A')),
                      (50, 0, 250, ('part_A.pt', '$p_T$ A [GeV]')))

    mB_vs_pTB   = H((50, 0, 50,  ('part_B.mass', 'Mass B [GeV]')),
                    (50, 0, 250, ('part_B.pt', '$p_T$ B [GeV]')))

    rhoB_vs_pTB   = H((50, 0, 1,  ('rhoB', 'rho B')),
                      (50, 0, 250, ('part_B.pt', '$p_T$ A [GeV]')))


    drAB      = H((100, 0, 5,   ('dr_AB', "$\Delta$ R AB")))

    zA        = H((100,  0.5, 2, ('zA', "z fraction")))
    zA_l      = H((100,  -3, 3, ('zA', "z fraction")))

    decay_phi = H((100, -0.1, 3.2, ('decay_phi', "decay angle")))
    thetaA    = H((100,  0, 2, ('thetaA',    "theta angle")))
    thetaA_l  = H((100,  0, 3.2, ('thetaA',    "theta angle")))

    tan_thetaA    = H((100,  0, 10, ('tan_thetaA',    "tan (theta angle)")))


    zA_vs_thetaA = H((50,  0.5, 1.5, ('zA', "z fraction")),
                     (50,  0, 1.5, ('thetaA',    "theta angle")))


    zA_vs_decay_phi = H((50,  0.5, 1.5, ('zA', "z fraction")),
                     (50,  -0.1, 3.2, ('decay_phi',    "decay angle")))

    thetaA_vs_decay_phi = H((50,  0, 1.5, ('thetaA',    "theta angle")),
                            (50,  -0.1, 3.2, ('decay_phi',    "decay angle")))


    zA_vs_pT = H((50,  0.5, 2, ('zA', "z fraction")),
                 (50,  50, 300, ('pt',    "pt")))

    thetaA_vs_pT = H((50,  0, 1.5, ('thetaA',    "theta angle")),
                     (50,  50, 300, ('pt',    "pt")))



    decay_phi_vs_pT = H((50 , -0.1, 3.2, ('decay_phi', "decay angle")),
                        (100,  50, 300, ('pt', "pT")))

    n         = H((0, 3,             ('n', 'Number')), n=ak.num)




    #
    #  For the PDFS
    #
    rhoA_pT = H((5, 50, 500, ("pt", "pT")),
                (50, 0, 0.5,  ('rhoA', 'rho A')))

    rhoB_pT = H((5, 50, 500, ("pt", "pT")),
                (50, 0, 1.0,  ('rhoB', 'rho B')))

    mA_pT = H((5, 50, 500, ("pt", "pT")),
                (50, 0, 50,  ('mA', 'mA [GeV]')))

    mB_pT = H((5, 50, 500, ("pt", "pT")),
                (50, 0, 50,  ('mB', 'mB [GeV]')))


    zA_pT = H((5, 50, 500, ("pt", "pT")),
              (50, 0.5, 1.3,  ('zA', 'z fraction')))

    thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                  (50, 0.0, 1.5,  ('thetaA', 'theta angle')))


    decay_phi_pT = H((5, 50, 500, ("pt", "pT")),
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))

    zA_vs_thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                        (50,  0.5, 1.3, ('zA', "z fraction")),
                        (50,  0, 1.5, ('thetaA',    "theta angle")))

    decay_phi_vs_eta_pT = H((5, 50, 500, ("pt", "pT")),
                            (50, 0, 3, ("abs_eta", "eta")),
                            (50 , -0.1, 3.2, ('decay_phi', "decay angle")))


    rhoA_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 0.5,  ('rhoA', 'rho A')))

    rhoB_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 0.5,  ('rhoB', 'rho B')))

    mA_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 50,  ('mA', 'mA [GeV]')))

    mB_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 50,  ('mB', 'mB [GeV]')))


    decay_phi_eta = H((5, 0, 3, ("abs_eta", "eta")),
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))


    zA_vs_thetaA_eta = H((5, 0, 3, ("abs_eta", "eta")),
                         (50,  0.5, 1.3, ('zA', "z fraction")),
                         (50,  0, 1.5, ('thetaA',    "theta angle")))
