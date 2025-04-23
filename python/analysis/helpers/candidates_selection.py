import numpy as np
import awkward as ak
from base_class.math.random import Squares
from analysis.helpers.SvB_helpers import compute_SvB


def create_cand_jet_dijet_quadjet(
    selev,
    apply_FvT: bool = False,
    run_SvB: bool = False,
    run_systematics: bool = False,
    classifier_SvB=None,
    classifier_SvB_MA=None,
    processOutput=None,
    isRun3=False,
):
    """
    Creates candidate jets, dijets, and quadjets for event selection.

    Parameters:
    -----------
    selev : ak.Array
        The selected events.
    apply_FvT : bool, optional
        Whether to apply FvT weights. Defaults to False.
    run_SvB : bool, optional
        Whether to run SvB classification. Defaults to False.
    run_systematics : bool, optional
        Whether to run systematics. Defaults to False.
    classifier_SvB : optional
        The SvB classifier. Defaults to None.
    classifier_SvB_MA : optional
        The SvB_MA classifier. Defaults to None.
    processOutput : optional
        Output dictionary for processing. Defaults to None.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.

    Returns:
    --------
    None
        Modifies the `selev` object in place.
    """
    #
    # To get test vectors
    #
    #jet_subset_dict = {key: getattr(selev.Jet,key)[0:10].tolist() for key in ["pt", "eta","phi", "mass","btagScore","bRegCorr","puId","jetId","selected", "selected_loose"]}
    #print(jet_subset_dict)

    #
    # Build and select boson candidate jets with bRegCorr applied
    #
    sorted_idx = ak.argsort( selev.Jet.btagScore * selev.Jet.selected, axis=1, ascending=False )
    canJet_idx = sorted_idx[:, 0:4]
    notCanJet_idx = sorted_idx[:, 4:]

    # # apply bJES to canJets
    canJet = selev.Jet[canJet_idx] * selev.Jet[canJet_idx].bRegCorr
    canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
    canJet["btagScore"] = selev.Jet.btagScore[canJet_idx]
    canJet["puId"] = selev.Jet.puId[canJet_idx]
    canJet["jetId"] = selev.Jet.jetId[canJet_idx]

    # CutFlow Debugging
    #if "pt_jec" in selev.Jet.fields:
    #    canJet["PNetRegPtRawCorr"] = selev.Jet.PNetRegPtRawCorr[canJet_idx]
    #    canJet["PNetRegPtRawCorrNeutrino"] = selev.Jet.PNetRegPtRawCorrNeutrino[canJet_idx]
    #    canJet["pt_raw"] = selev.Jet.pt_raw[canJet_idx]

    if "hadronFlavour" in selev.Jet.fields:
        canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]

    #
    # pt sort canJets
    #
    canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
    selev["canJet"] = canJet
    for i in range(4):
        selev[f"canJet{i}"] = selev["canJet"][:, i]

    selev["v4j"] = canJet.sum(axis=1)
    notCanJet = selev.Jet[notCanJet_idx]
    notCanJet = notCanJet[notCanJet.selected_loose]
    notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

    notCanJet["isSelJet"] = 1 * ( (notCanJet.pt >= 40) & (np.abs(notCanJet.eta) < 2.4) )
    selev["notCanJet_coffea"] = notCanJet
    selev["nNotCanJet"] = ak.num(selev.notCanJet_coffea)

    # Build diJets, indexed by diJet[event,pairing,0/1]
    canJet = selev["canJet"]
    pairing = [([0, 2], [0, 1], [0, 1]), ([1, 3], [2, 3], [3, 2])]
    diJet = canJet[:, pairing[0]] + canJet[:, pairing[1]]
    diJet["lead"] = canJet[:, pairing[0]]
    diJet["subl"] = canJet[:, pairing[1]]
    diJet["st"] = diJet["lead"].pt + diJet["subl"].pt
    # diJet["mass"] = (diJet["lead"] + diJet["subl"]).mass
    diJet["dr"] = diJet["lead"].delta_r(diJet["subl"])
    diJet["dphi"] = diJet["lead"].delta_phi(diJet["subl"])

    # Sort diJets within views to be lead st, subl st
    if isRun3:
        diJet = diJet[ak.argsort(diJet.pt, axis=2, ascending=False)]
    else:
        diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
    diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]
    # Now indexed by diJet[event,pairing,lead/subl st]

    # Compute diJetMass cut with independent min/max for lead/subl
    minDiJetMass = np.array([[[52, 50]]])
    maxDiJetMass = np.array([[[180, 173]]])
    diJet["passDiJetMass"] = (minDiJetMass < diJet.mass) & ( diJet.mass < maxDiJetMass )

    # Compute MDRs
    min_m4j_scale = np.array([[360, 235]])
    min_dr_offset = np.array([[-0.5, 0.0]])
    max_m4j_scale = np.array([[650, 650]])
    max_dr_offset = np.array([[0.5, 0.7]])
    max_dr = np.array([[1.5, 1.5]])
    # m4j = np.repeat(np.reshape(np.array(selev["v4j"].mass), (-1, 1, 1)), 2, axis=2)
    m4j = selev["v4j"].mass[:, np.newaxis, np.newaxis]
    diJet["passMDR"] = (min_m4j_scale / m4j + min_dr_offset < diJet.dr) & ( diJet.dr < np.maximum(max_m4j_scale / m4j + max_dr_offset, max_dr) )

    #
    # Compute consistency of diJet masses with boson masses
    #
    mZ = 91.0
    mH = 125.0
    st_bias = np.array([[[1.02, 0.98]]])
    cZ = mZ * st_bias
    cH = mH * st_bias

    diJet["xZ"] = (diJet.mass - cZ) / (0.1 * diJet.mass)
    diJet["xH"] = (diJet.mass - cH) / (0.1 * diJet.mass)

    #
    # Build quadJets
    #
    rng_0 = Squares("quadJetSelection")
    rng_1 = rng_0.shift(1)
    rng_2 = rng_0.shift(2)
    counter = selev.event

    # print(f"{self.chunk} mass {diJet[:, :, 0].mass[0:5]}\n")
    # print(f"{self.chunk} mass view64 {np.asarray(diJet[:, :, 0].mass).view(np.uint64)[0:5]}\n")
    # print(f"{self.chunk} mass rounded view64 {np.round(np.asarray(diJet[:, :, 0].mass), 0).view(np.uint64)[0:5]}\n")
    # print(f"{self.chunk} mass rounded {np.round(np.asarray(diJet[:, :, 0].mass), 0)[0:5]}\n")

    quadJet = ak.zip( { "lead": diJet[:, :, 0],
                        "subl": diJet[:, :, 1],
                        "close": diJetDr[:, :, 0],
                        "other": diJetDr[:, :, 1],
                        "passDiJetMass": ak.all(diJet.passDiJetMass, axis=2),
                        "random": np.concatenate([rng_0.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
                                                  rng_1.uniform(counter, low=0.1, high=0.9)[:, np.newaxis],
                                                  rng_2.uniform(counter, low=0.1, high=0.9)[:, np.newaxis]], axis=1),

                       } )

    quadJet["dr"] = quadJet["lead"].delta_r(quadJet["subl"])
    quadJet["dphi"] = quadJet["lead"].delta_phi(quadJet["subl"])
    quadJet["deta"] = quadJet["lead"].eta - quadJet["subl"].eta
    quadJet["v4jmass"] = selev["v4j"].mass

    #
    # Compute Signal Regions
    #
    quadJet["xZZ"] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
    quadJet["xHH"] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
    quadJet["xZH"] = np.sqrt( np.minimum( quadJet.lead.xH**2 + quadJet.subl.xZ**2, quadJet.lead.xZ**2 + quadJet.subl.xH**2, ) )

    max_xZZ = 2.6
    max_xZH = 1.9
    max_xHH = 1.9
    quadJet["ZZSR"] = quadJet.xZZ < max_xZZ
    quadJet["ZHSR"] = quadJet.xZH < max_xZH
    quadJet["HHSR"] = ((quadJet.xHH < max_xHH) & selev.notInBoostedSel ) if 'notInBoostedSel' in selev.fields else (quadJet.xHH < max_xHH)  ## notInBoostedSel is true by default


    if isRun3:

        # Compute distances to diagonal
        #   https://gitlab.cern.ch/mkolosov/hh4b_run3/-/blob/run2/python/producers/hh4bTreeProducer.py#L3386
        diagonalXoYo = 1.04
        quadJet["dhh"] = (1.0/np.sqrt(1+pow(diagonalXoYo, 2)))*abs(quadJet["lead"].mass - ((diagonalXoYo)*quadJet["subl"].mass))
        quadJet["selected"] = quadJet.dhh == np.min(quadJet.dhh, axis=1)


        #
        #   For CR selection
        #
        cLead = 125
        cSubl = 120
        SR_radius = 30
        CR_radius = 55
        quadJet["selected"] = quadJet.dhh == np.min(quadJet.dhh, axis=1)

        quadJet["rhh"] = np.sqrt( (quadJet["lead"].mass - cLead)**2 + (quadJet["subl"].mass - cSubl)**2 )
        quadJet["SR"] = (quadJet.rhh < SR_radius)
        quadJet["SB"] =  (~quadJet.SR) & (quadJet.rhh < CR_radius)
        quadJet["passDiJetMass"] =  quadJet.SR | quadJet.SB

    else:

        quadJet["SR"] = quadJet.ZZSR | quadJet.ZHSR | quadJet.HHSR
        quadJet["SB"] = quadJet.passDiJetMass & ~quadJet.SR

        #
        # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
        #
        quadJet["rank"] = ( 10 * quadJet.passDiJetMass + quadJet.lead.passMDR + quadJet.subl.passMDR + quadJet.random )
        quadJet["selected"] = quadJet.rank == np.max(quadJet.rank, axis=1)


    if apply_FvT:
        quadJet["FvT_q_score"] = np.concatenate( [
            selev.FvT.q_1234[:, np.newaxis],
            selev.FvT.q_1324[:, np.newaxis],
            selev.FvT.q_1423[:, np.newaxis],
        ], axis=1, )

    if run_SvB:

        if (classifier_SvB is not None) | (classifier_SvB_MA is not None):

            if run_systematics: tmp_mask = (selev.fourTag & quadJet[quadJet.selected][:, 0].SR)
            else: tmp_mask = np.full(len(selev), True)
            compute_SvB(selev,
                        tmp_mask,
                        SvB=classifier_SvB,
                        SvB_MA=classifier_SvB_MA,
                        doCheck=False)

        quadJet["SvB_q_score"] = np.concatenate( [
            selev.SvB.q_1234[:, np.newaxis],
            selev.SvB.q_1324[:, np.newaxis],
            selev.SvB.q_1423[:, np.newaxis],
            ], axis=1, )

        quadJet["SvB_MA_q_score"] = np.concatenate( [
            selev.SvB_MA.q_1234[:, np.newaxis],
            selev.SvB_MA.q_1324[:, np.newaxis],
            selev.SvB_MA.q_1423[:, np.newaxis],
            ], axis=1, )

    selev["diJet"] = diJet
    selev["quadJet"] = quadJet
    selev["quadJet_selected"] = quadJet[quadJet.selected][:, 0]
    selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)
    #
    #  Build the close dR and other quadjets
    #    (There is Probably a better way to do this ...
    #
    arg_min_close_dr = np.argmin(quadJet.close.dr, axis=1)
    arg_min_close_dr = arg_min_close_dr.to_numpy()
    selev["quadJet_min_dr"] = quadJet[ np.array(range(len(quadJet))), arg_min_close_dr ]


    selev["m4j"] = selev.v4j.mass
    selev["m4j_HHSR"] = ak.where(~selev.quadJet_selected.HHSR, -2, selev.m4j)
    selev["m4j_ZHSR"] = ak.where(~selev.quadJet_selected.ZHSR, -2, selev.m4j)
    selev["m4j_ZZSR"] = ak.where(~selev.quadJet_selected.ZZSR, -2, selev.m4j)

    selev['leadStM_selected'] = selev.quadJet_selected.lead.mass
    selev['sublStM_selected'] = selev.quadJet_selected.subl.mass

    selev['dijet_HHSR'] = ak.zip( { "lead_m": ak.where(~selev.quadJet_selected.HHSR, -2, selev.leadStM_selected),
                                    "subl_m": ak.where(~selev.quadJet_selected.HHSR, -2, selev.sublStM_selected),
                                } )
    selev['dijet_ZHSR'] = ak.zip( { "lead_m": ak.where(~selev.quadJet_selected.ZHSR, -2, selev.leadStM_selected),
                                    "subl_m": ak.where(~selev.quadJet_selected.ZHSR, -2, selev.sublStM_selected),
                                    } )
    selev['dijet_ZZSR'] = ak.zip( { "lead_m": ak.where(~selev.quadJet_selected.ZZSR, -2, selev.leadStM_selected),
                                    "subl_m": ak.where(~selev.quadJet_selected.ZZSR, -2, selev.sublStM_selected),
                                    } )

    selev["region"] = ak.zip({
        "SR": selev["quadJet_selected"].SR,
        "SB": selev["quadJet_selected"].SB
        })

    #
    # Debugging the skimmer
    #
    ### selev_mask = selev.event == 434011
    ### out_data = {}
    ### out_data["debug_event"  ]            = selev.event[selev_mask]
    ### out_data["debug_qj_rank"  ]    = quadJet[selev_mask].rank.to_list()
    ### out_data["debug_qj_selected"  ]    = quadJet[selev_mask].selected.to_list()
    ### out_data["debug_qj_passDiJetMass"  ]    = quadJet[selev_mask].passDiJetMass.to_list()
    ### out_data["debug_qj_lead_passMDR"  ]    = quadJet[selev_mask].lead.passMDR.to_list()
    ### out_data["debug_qj_subl_passMDR"  ]    = quadJet[selev_mask].subl.passMDR.to_list()
    ### out_data["debug_qj_lead_mass"  ]    = quadJet[selev_mask].lead.mass.to_list()
    ### out_data["debug_qj_subl_mass"  ]    = quadJet[selev_mask].subl.mass.to_list()
    ### out_data["debug_qj_random"  ]    = quadJet[selev_mask].random.to_list()
    ### out_data["debug_qj_SR"  ]    = quadJet[selev_mask].SR.to_list()
    ### out_data["debug_qj_HHSR"  ]    = quadJet[selev_mask].HHSR.to_list()
    ### out_data["debug_qj_ZZSR"  ]    = quadJet[selev_mask].ZZSR.to_list()
    ### out_data["debug_qj_ZHSR"  ]    = quadJet[selev_mask].ZHSR.to_list()
    ### out_data["debug_qj_xZZ"  ]    = quadJet[selev_mask].xZZ.to_list()
    ### out_data["debug_qj_xZH"  ]    = quadJet[selev_mask].xZH.to_list()
    ### out_data["debug_qj_xHH"  ]    = quadJet[selev_mask].xHH.to_list()
    ### out_data["debug_qj_ZHSR"  ]    = quadJet[selev_mask].ZHSR.to_list()
    ### out_data["debug_qj_lead_xZ"  ]    = quadJet[selev_mask].lead.xZ.to_list()
    ### out_data["debug_qj_lead_xH"  ]    = quadJet[selev_mask].lead.xH.to_list()
    ### out_data["debug_qj_subl_xZ"  ]    = quadJet[selev_mask].subl.xZ.to_list()
    ### out_data["debug_qj_subl_xH"  ]    = quadJet[selev_mask].subl.xH.to_list()
    ### out_data["debug_qj_SB"  ]    = quadJet[selev_mask].SB.to_list()
    ### out_data["debug_counter"  ]    = counter[selev_mask].to_list()
    ### out_data["debug_SR"] = selev["quadJet_selected"][selev_mask].SR
    ### out_data["debug_SB"] = selev["quadJet_selected"][selev_mask].SB
    ### out_data["debug_threeTag"] = selev[selev_mask].threeTag
    ### out_data["debug_fourTag"] = selev[selev_mask].fourTag
    ### out_data["debug_qj_lead_pt"  ]         = quadJet[selev_mask].lead.pt.to_list()
    ### out_data["debug_qj_lead_lead_pt"  ]    = quadJet[selev_mask].lead.lead.pt.to_list()
    ### out_data["debug_qj_lead_lead_eta"  ]   = quadJet[selev_mask].lead.lead.eta.to_list()
    ### out_data["debug_qj_lead_lead_phi"  ]   = quadJet[selev_mask].lead.lead.phi.to_list()
    ### out_data["debug_qj_lead_lead_mass"  ]  = quadJet[selev_mask].lead.lead.mass.to_list()
    ### out_data["debug_qj_lead_subl_pt"  ]    = quadJet[selev_mask].lead.subl.pt.to_list()
    ### out_data["debug_qj_lead_subl_eta"  ]   = quadJet[selev_mask].lead.subl.eta.to_list()
    ### out_data["debug_qj_lead_subl_phi"  ]   = quadJet[selev_mask].lead.subl.phi.to_list()
    ### out_data["debug_qj_lead_subl_mass"  ]  = quadJet[selev_mask].lead.subl.mass.to_list()
    ###
    ### out_data["debug_qj_subl_pt"  ]         = quadJet[selev_mask].subl.pt.to_list()
    ### out_data["debug_qj_subl_lead_pt"  ]    = quadJet[selev_mask].subl.lead.pt.to_list()
    ### out_data["debug_qj_subl_lead_eta"  ]   = quadJet[selev_mask].subl.lead.eta.to_list()
    ### out_data["debug_qj_subl_lead_phi"  ]   = quadJet[selev_mask].subl.lead.phi.to_list()
    ### out_data["debug_qj_subl_lead_mass"  ]  = quadJet[selev_mask].subl.lead.mass.to_list()
    ###
    ### out_data["debug_qj_subl_subl_pt"  ]    = quadJet[selev_mask].subl.subl.pt.to_list()
    ### out_data["debug_qj_subl_subl_eta"  ]   = quadJet[selev_mask].subl.subl.eta.to_list()
    ### out_data["debug_qj_subl_subl_phi"  ]   = quadJet[selev_mask].subl.subl.phi.to_list()
    ### out_data["debug_qj_subl_subl_mass"  ]  = quadJet[selev_mask].subl.subl.mass.to_list()
    ###
    ###
    ### for out_k, out_v in out_data.items():
    ###     processOutput[out_k] = {}
    ###     processOutput[out_k][selev.metadata['dataset']] = list(out_v)



    if run_SvB:
        selev["passSvB"] = selev["SvB_MA"].ps > 0.80
        selev["failSvB"] = selev["SvB_MA"].ps < 0.05