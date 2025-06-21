import awkward as ak
from base_class.physics.object import LorentzVector, Jet
from base_class.hist import H, Template

class ClusterHists(Template):
    pt        = H((100,  0, 300, ('pt',    "pt [GeV]")))
    pt_l      = H((100,  0, 500, ('pt',    "pt [GeV]")))

    mA        = H((100, 0, 100,  ('mA', "mA [GeV]")))
    mA_l      = H((100, 0, 400,  ('mA', "mA [GeV]")))
    mA_vl     = H((100, 0, 1000, ('mA', "mA [GeV]")))

    mB        = H((100, 0,  60,  ('mB', "mB [GeV]")))
    mB_l      = H((100, 0, 400,  ('mB', "mB [GeV]")))
    mB_vl     = H((100, 0, 600,  ('mB', "mB [GeV]")))

    zA        = H((100,  0.5, 1.3, ('zA', "z fraction")))
    zA_l      = H((100,  0, 1.5, ('zA', "z fraction")))
    zA_vl      = H((100,  -3, 3, ('zA', "z fraction")))

    decay_phi = H((100, -0.1, 3.2, ('decay_phi', "decay angle")))
    thetaA    = H((100,  0, 1.5, ('thetaA',    "theta angle")))

    n         = H((0, 3,             ('n', 'Number')), n=ak.num)

    mA_rot    = H((100, 0, 100,  ('mA_rotated', "mA [GeV]")))
    mB_rot    = H((100, 0, 60,   ('mB_rotated', "mB [GeV]")))


    #
    #  For the PDFS
    #

    mA_pT = H((5, 50, 500, ("pt", "pT")),
              (100, 0, 100,  ('mA', 'mA [GeV]')))

    mB_pT = H((5, 50, 500, ("pt", "pT")),
              (100, 0, 60,  ('mB', 'mB [GeV]')))

    mA_r_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 100,  ('mA_rotated', 'mA [GeV]')))

    mB_r_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 60,  ('mB_rotated', 'mB [GeV]')))


    mA_l_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 400,  ('mA', 'mA [GeV]')))

    mB_l_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 400,  ('mB', 'mB [GeV]')))


    mA_vl_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 1000,  ('mA', 'mA [GeV]')))

    mB_vl_pT = H((5, 50, 500, ("pt", "pT")),
                (100, 0, 600,  ('mB', 'mB [GeV]')))


    decay_phi_pT = H((5, 50, 500, ("pt", "pT")),
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))

    zA_vs_thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                        (50,  0.5, 1.3, ('zA', "z fraction")),
                        (50,  0, 1.5, ('thetaA',    "theta angle")))

    zA_l_vs_thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                          (50,  0, 1.5, ('zA', "z fraction")),
                          (50,  0, 1.5, ('thetaA',    "theta angle")))


    rhoA_pT = H((5, 50, 500, ("pt", "pT")),
              (50, 0, 0.5,  ('rhoA', 'rhoA (mass/pt)')))

    rhoB_pT = H((5, 50, 500, ("pt", "pT")),
                (50, 0, 0.5,  ('rhoB', 'rhoB (mass/pt)')))


class ClusterHistsBoosted(Template):
    pt_l      = H((100,  0, 1000, ('pt',    "pt [GeV]")))

    mA        = H((100, 0, 100,  ('mA', "mA [GeV]")))
    mA_l      = H((100, 0, 400,  ('mA', "mA [GeV]")))
    mA_vl     = H((100, 0, 1000, ('mA', "mA [GeV]")))

    mB        = H((100, 0,  60,  ('mB', "mB [GeV]")))
    mB_l      = H((100, 0, 400,  ('mB', "mB [GeV]")))
    mB_vl     = H((100, 0, 600,  ('mB', "mB [GeV]")))

    zA        = H((100,  0.5, 1.3, ('zA', "z fraction")))
    zA_l      = H((100,  0, 1.5, ('zA', "z fraction")))
    zA_vl      = H((100,  -3, 3, ('zA', "z fraction")))

    decay_phi = H((100, -0.1, 3.2, ('decay_phi', "decay angle")))
    thetaA    = H((100,  0, 1.5, ('thetaA',    "theta angle")))

    n         = H((0, 3,             ('n', 'Number')), n=ak.num)

    mA_rot    = H((100, 0, 100,  ('mA_rotated', "mA [GeV]")))
    mB_rot    = H((100, 0, 60,   ('mB_rotated', "mB [GeV]")))


    #
    #  For the PDFS
    #

    mA_pT = H((3, 300, 1000, ("pt", "pT")),
              (100, 0, 100,  ('mA', 'mA [GeV]')))

    mB_pT = H((3, 300, 1000, ("pt", "pT")),
              (100, 0, 60,  ('mB', 'mB [GeV]')))

    mA_r_pT = H((3, 300, 1000, ("pt", "pT")),
                (100, 0, 100,  ('mA_rotated', 'mA [GeV]')))

    mB_r_pT = H((3, 300, 1000, ("pt", "pT")),
                (100, 0, 60,  ('mB_rotated', 'mB [GeV]')))


    mA_l_pT = H((3, 300, 1000, ("pt", "pT")),
                (100, 0, 400,  ('mA', 'mA [GeV]')))

    mB_l_pT = H((3, 300, 1000, ("pt", "pT")),
                (100, 0, 400,  ('mB', 'mB [GeV]')))


    mA_vl_pT = H((3, 300, 1000, ("pt", "pT")),
                (100, 0, 1000,  ('mA', 'mA [GeV]')))

    mB_vl_pT = H((3, 300, 1000, ("pt", "pT")),
                (100, 0, 600,  ('mB', 'mB [GeV]')))


    decay_phi_pT = H((3, 300, 1000, ("pt", "pT")),
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))

    zA_vs_thetaA_pT = H((3, 300, 1000, ("pt", "pT")),
                        (50,  0.5, 1.3, ('zA', "z fraction")),
                        (50,  0, 1.5, ('thetaA',    "theta angle")))

    zA_l_vs_thetaA_pT = H((3, 300, 1000, ("pt", "pT")),
                          (50,  0, 1.5, ('zA', "z fraction")),
                          (50,  0, 1.5, ('thetaA',    "theta angle")))


    rhoA_pT = H((3, 300, 1000, ("pt", "pT")),
              (50, 0, 0.5,  ('rhoA', 'rhoA (mass/pt)')))

    rhoB_pT = H((3, 300, 1000, ("pt", "pT")),
                (50, 0, 0.5,  ('rhoB', 'rhoB (mass/pt)')))





class ClusterHistsDetailed(ClusterHists):
    dpt_AB        = H((50,  -50, 50, ('dpt_AB',    "pt [GeV]")))

    pt_A      = H((100,  0, 300, ('part_A.pt',    "pt [GeV]")))
    pt_B      = H((100,  0, 300, ('part_B.pt',    "pt [GeV]")))

    zA_pT = H((5, 50, 500, ("pt", "pT")),
              (50, 0.5, 1.3,  ('zA', 'z fraction')))

    thetaA_pT = H((5, 50, 500, ("pt", "pT")),
                  (50, 0.0, 1.5,  ('thetaA', 'theta angle')))


    pz        = H((100, -500, 500, ('pz',    "pz [GeV]")))
    eta        = H((50,  -3, 3, ('eta',    "eta")))

    rhoA      = H((100, 0, 0.5,  ('rhoA', "rho A (mass/pt)")))
    rhoB      = H((100, 0, 1,  ('rhoB', "rho B (mass/pt)")))

    mA_vs_mB   = H((50, 0, 50, ('part_A.mass', 'Mass A [GeV]')),
                  (50, 0, 50, ('part_B.mass', 'Mass A [GeV]')))

    mA_vs_pTA   = H((50, 0, 50,  ('part_A.mass', 'Mass A [GeV]')),
                    (50, 0, 250, ('part_A.pt', '$p_T$ A [GeV]')))

    rhoA_vs_pTA   = H((50, 0, 1,  ('rhoA', 'rho A')),
                      (50, 0, 250, ('part_A.pt', '$p_T$ A [GeV]')))

    mB_vs_pTB   = H((50, 0, 50,  ('part_B.mass', 'Mass B [GeV]')),
                    (50, 0, 250, ('part_B.pt', '$p_T$ B [GeV]')))

    rhoB_vs_pTB   = H((50, 0, 1,  ('rhoB', 'rho B')),
                      (50, 0, 250, ('part_B.pt', '$p_T$ B [GeV]')))

    drAB      = H((100, 0, 5,   ('dr_AB', "$\Delta$ R AB")))
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


    mA_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 50,  ('mA', 'mA [GeV]')))

    mB_eta = H((5, 0, 3, ("abs_eta", "eta")),
                (50, 0, 50,  ('mB', 'mB [GeV]')))


    decay_phi_eta = H((5, 0, 3, ("abs_eta", "eta")),
                     (50 , -0.1, 3.2, ('decay_phi', "decay angle")))


    zA_vs_thetaA_eta = H((5, 0, 3, ("abs_eta", "eta")),
                         (50,  0.5, 1.3, ('zA', "z fraction")),
                         (50,  0, 1.5, ('thetaA',    "theta angle")))
