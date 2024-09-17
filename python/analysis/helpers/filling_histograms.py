from analysis.helpers.hist_templates import (
    FvTHists,
    QuadJetHists,
    QuadJetHistsSRSingle,
    SvBHists,
    TopCandHists,
    WCandHists,
)
from base_class.hist import Collection, Fill
from base_class.physics.object import Elec, Jet, LorentzVector, Muon
import logging

def filling_nominal_histograms(selev, JCM,
                               processName: str = None, 
                               year: str = 'UL18', 
                               isMC: bool = False, 
                               histCuts: list = [], 
                               apply_FvT: bool = False, 
                               run_SvB: bool = False, 
                               top_reconstruction: bool = False, 
                               isMixedData: bool = False, 
                               isDataForMixed: bool = False, 
                               isTTForMixed: bool = False, 
                               event_metadata: dict = {},
                               ):

    fill = Fill(process=processName, year=year, weight="weight")

    hist = Collection( process=[processName],
                        year=[year],
                        tag=[3, 4, 0],  # 3 / 4/ Other
                        region=[2, 1, 0],  # SR / SB / Other
                        **dict((s, ...) for s in histCuts)
                        )

    #
    # To Add
    #

    #    m4j_vs_leadSt_dR = dir.make<TH2F>("m4j_vs_leadSt_dR", (name+"/m4j_vs_leadSt_dR; m_{4j} [GeV]; S_{T} leading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);
    #    m4j_vs_sublSt_dR = dir.make<TH2F>("m4j_vs_sublSt_dR", (name+"/m4j_vs_sublSt_dR; m_{4j} [GeV]; S_{T} subleading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);

    fill += hist.add( "trigWeight", (40, 0, 2, ("trigWeight.Data", 'Trigger weight')), weight='no_weight' )

    fill += hist.add( "nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")) )
    fill += hist.add( "nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")), )

    fill += hist.add( "hT", (50, 0, 1500, ("hT", "h_{T} [GeV]")) )
    fill += hist.add( "hT_selected", (50, 0, 1500, ("hT_selected", "h_{T} [GeV]")), )

    fill += hist.add("xW",  (100, -12, 12, ("xW", "xW")))
    fill += hist.add("xbW", (100, -15, 15, ("xbW", "xbW")))

    #
    # Separate reweighting for the different mixed samples
    #
    if isDataForMixed:
        for _FvT_name in event_metadata["FvT_names"]:
            fill += SvBHists( (f"SvB_{_FvT_name}",    "SvB Classifier"),    "SvB",    weight=f"weight_{_FvT_name}" )
            fill += SvBHists( (f"SvB_MA_{_FvT_name}", "SvB MA Classifier"), "SvB_MA", weight=f"weight_{_FvT_name}" )

    #
    # Jets
    #
    fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])
    fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=["deepjet_c"])
    fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=["deepjet_c"])
    fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=["deepjet_c"])

    #
    #  Make quad jet hists
    #
    fill += LorentzVector.plot_pair( ("v4j", R"$HH_{4b}$"), "v4j", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )
    fill += QuadJetHists( ("quadJet_selected", "Selected Quad Jet"), "quadJet_selected" )
    fill += QuadJetHists( ("quadJet_min_dr", "Min dR Quad Jet"), "quadJet_min_dr" )
    fill += hist.add( "m4j_HHSR", (120, 0, 1200, ("m4j_HHSR", "m4j HHSR")) )
    fill += hist.add( "m4j_ZHSR", (120, 0, 1200, ("m4j_ZHSR", "m4j ZHSR")) )
    fill += hist.add( "m4j_ZZSR", (120, 0, 1200, ("m4j_ZZSR", "m4j ZZSR")) )

    fill += QuadJetHistsSRSingle( ("dijet_HHSR", "DiJet Mass HHSR") ,"dijet_HHSR"  )
    fill += QuadJetHistsSRSingle( ("dijet_ZHSR", "DiJet Mass ZHSR") ,"dijet_ZHSR"  )
    fill += QuadJetHistsSRSingle( ("dijet_ZZSR", "DiJet Mass ZZSR") ,"dijet_ZZSR"  )

    #
    #  Make classifier hists
    #
    if apply_FvT:
        FvT_skip = []
        if isMixedData or isDataForMixed or isTTForMixed:
            FvT_skip = ["pt", "pm3", "pm4"]

        fill += FvTHists(("FvT", "FvT Classifier"), "FvT", skip=FvT_skip)

        fill += hist.add("quadJet_selected_FvT_score", (100, 0, 1, ("quadJet_selected.FvT_q_score", "Selected Quad Jet Diboson FvT q score") ) )
        fill += hist.add("quadJet_min_FvT_score",      (100, 0, 1, ("quadJet_min_dr.FvT_q_score",   "Min dR Quad Jet Diboson FvT q score"  ) ) )

        if JCM:
            fill += hist.add("FvT_noFvT", (100, 0, 5, ("FvT.FvT", "FvT reweight")), weight="weight_noFvT")

    skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]

    fill += Jet.plot( ("selJets_noJCM", "Selected Jets"),        "selJet",       weight="weight_noJCM_noFvT", skip=skip_all_but_n, )
    fill += Jet.plot( ("tagJets_noJCM", "Tag Jets"),             "tagJet",       weight="weight_noJCM_noFvT", skip=skip_all_but_n, )
    fill += Jet.plot( ("tagJets_loose_noJCM", "Loose Tag Jets"), "tagJet_loose", weight="weight_noJCM_noFvT", skip=skip_all_but_n, )

    for iJ in range(4):
        fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )

    #
    #  Leptons
    #
    skip_muons = ["charge"] + Muon.skip_detailed_plots
    if not isMC:
        skip_muons += ["genPartFlav"]
    fill += Muon.plot( ("selMuons", "Selected Muons"), "selMuon", skip=skip_muons )

    if not isMixedData:
        skip_elecs = ["charge"] + Elec.skip_detailed_plots
        if not isMC:
            skip_elecs += ["genPartFlav"]
        fill += Elec.plot( ("selElecs", "Selected Elecs"), "selElec", skip=skip_elecs )

    #
    # Top Candidates
    #
    if top_reconstruction in ["slow","fast"]:
        fill += TopCandHists(("top_cand", "Top Candidate"), "top_cand")

    if run_SvB:

        fill += SvBHists(("SvB",    "SvB Classifier"),    "SvB")
        fill += SvBHists(("SvB_MA", "SvB MA Classifier"), "SvB_MA")
        fill += hist.add( "quadJet_selected_SvB_q_score", ( 100, 0, 1, ( "quadJet_selected.SvB_q_score",  "Selected Quad Jet Diboson SvB q score") ) )
        fill += hist.add( "quadJet_min_SvB_MA_q_score",   ( 100, 0, 1, ( "quadJet_min_dr.SvB_MA_q_score", "Min dR Quad Jet Diboson SvB MA q score") ) )
        if isDataForMixed:
            for _FvT_name in event_metadata["FvT_names"]:
                fill += SvBHists( (f"SvB_{_FvT_name}",    "SvB Classifier"),    "SvB",    weight=f"weight_{_FvT_name}", )
                fill += SvBHists( (f"SvB_MA_{_FvT_name}", "SvB MA Classifier"), "SvB_MA", weight=f"weight_{_FvT_name}", )

    #
    # fill histograms
    #
    # fill.cache(selev)
    fill(selev, hist)

    return hist.output


def filling_syst_histograms(selev, weights, analysis_selections,  
                            shift_name: str = 'nominal', 
                            processName: str = None, 
                            year: str = 'UL18', 
                            histCuts: list = []
                            ):

    shift_name = "nominal" if not shift_name else shift_name
    hist_SvB = Collection( process=[processName],
                            year=[year],
                            variation=[shift_name],
                            tag=[4],  # 3 / 4/ Other
                            region=[2],  # SR / SB / Other
                            **dict((s, ...) for s in histCuts),
                            )

    fill_SvB = Fill( process=processName, year=year, variation=shift_name, weight="weight" )
    fill_SvB += SvBHists(("SvB",    "SvB Classifier"),    "SvB",    skip=["ps", "ptt"])
    fill_SvB += SvBHists(("SvB_MA", "SvB MA Classifier"), "SvB_MA", skip=["ps", "ptt"])

    fill_SvB(selev, hist_SvB)

    if "nominal" in shift_name:
        logging.info(f"Weight variations {weights.variations}")

        dict_hist_SvB = {}
        for ivar in list(weights.variations):

            dict_hist_SvB[ivar] = Collection( process=[processName],
                                                year=[year],
                                                variation=[ivar],
                                                tag=[4],  # 3 / 4/ Other
                                                region=[2],  # SR / SB / Other
                                                **dict((s, ...) for s in histCuts) )

            selev[f"weight_{ivar}"] = weights.weight(modifier=ivar)[ analysis_selections ]
            fill_SvB_ivar = Fill( process=processName, year=year, variation=ivar, weight=f"weight_{ivar}", )

            logging.debug(f"{ivar} {selev['weight']}")

            fill_SvB_ivar += SvBHists( ("SvB",    "SvB Classifier"),    "SvB",    skip=["ps", "ptt"] )
            fill_SvB_ivar += SvBHists( ("SvB_MA", "SvB MA Classifier"), "SvB_MA", skip=["ps", "ptt"] )

            fill_SvB_ivar(selev, dict_hist_SvB[ivar])

            for ih in hist_SvB.output["hists"].keys():
                hist_SvB.output["hists"][ih] = ( hist_SvB.output["hists"][ih] + dict_hist_SvB[ivar].output["hists"][ih] )

    return hist_SvB.output