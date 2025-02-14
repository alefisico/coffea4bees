import numpy as np
import awkward as ak
from analysis.helpers.common import (
    mask_event_decision,
    drClean,
    apply_jet_veto_maps,
    apply_jerc_corrections,
    compute_puid
)
from analysis.helpers.SvB_helpers import compute_SvB
from coffea.lumi_tools import LumiMask
from base_class.math.random import Squares
from copy import copy
import logging

def apply_event_selection_4b( event, corrections_metadata, *, cut_on_lumimask=True):

    lumimask = LumiMask(corrections_metadata['goldenJSON'])
    event['lumimask'] = np.array( lumimask(event.run, event.luminosityBlock) ) \
            if cut_on_lumimask else np.full(len(event), True)

    event['passHLT'] = np.full(len(event), True) \
            if ('HLT' not in event.fields) else mask_event_decision( event,
                    decision="OR", branch="HLT", list_to_mask=event.metadata['trigger']  )

    event['passNoiseFilter'] = np.full(len(event), True) \
            if 'Flag' not in event.fields else mask_event_decision( event,
                    decision="AND", branch="Flag",
                    list_to_mask=corrections_metadata['NoiseFilter'],
                    list_to_skip=['BadPFMuonDzFilter', 'hfNoisyHitsFilter']  )

    return event

def apply_object_selection_4b(event, corrections_metadata, *,
                              dataset: str = '',
                              doLeptonRemoval: bool = True,
                              loosePtForSkim: bool = False,
                              override_selected_with_flavor_bit: bool = False,
                              run_lowpt_selection: bool = False,
                              do_jet_veto_maps: bool = False,
                              isRun3: bool = False,
                              isMC: bool = False,  ### temporary for Run3
                              isSyntheticData: bool = False,
                              ):
    """docstring for apply_basic_selection_4b. This fuction is not modifying the content of anything in events. it is just adding it"""

    #
    # Combined RunII and 3 selection
    #

    #
    # Adding muons (loose muon id definition)
    # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
    #
    event['Muon', 'selected'] = (event.Muon.pt > 10) & (abs(event.Muon.eta) < 2.5) & (event.Muon.pfRelIso04_all < 0.15) & (event.Muon.looseId)
    event['nMuon_selected'] = ak.sum(event.Muon.selected, axis=1)
    event['selMuon'] = event.Muon[event.Muon.selected]

    # Adding electrons (loose electron id)
    # https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
    #
    if 'Electron' in event.fields:

        if isRun3:
            event['Electron', 'selected'] = (event.Electron.pt > 15) & (abs(event.Electron.eta) < 2.5) & (event.Electron.pfRelIso03_all < 0.15) & getattr(event.Electron, 'mvaNoIso_WP90' )
        else:
            event['Electron', 'selected'] = (event.Electron.pt > 15) & (abs(event.Electron.eta) < 2.5) & (event.Electron.pfRelIso03_all < 0.15) & getattr(event.Electron, 'mvaFall17V2Iso_WP90' )

        event['nElectron_selected'] = ak.sum(event.Electron.selected, axis=1)
        event['selElec'] = event.Electron[event.Electron.selected]
        selLepton = ak.concatenate( [event.selElec, event.selMuon], axis=1 )
    else: selLepton = event.selMuon


    if doLeptonRemoval:
        event['Jet', 'lepton_cleaned'] = drClean( event.Jet, selLepton )[1]  ### 0 is the collection of jets, 1 is the flag
    else:
        event['Jet', 'lepton_cleaned'] = np.full(len(event), True)

    # For trigger emulation
    event['Jet', 'muon_cleaned'] = drClean( event.Jet, event.selMuon )[1]
    event['Jet', 'ht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) & event.Jet.muon_cleaned

    if do_jet_veto_maps:
        event['Jet', 'jet_veto_maps'] = apply_jet_veto_maps( corrections_metadata['jet_veto_maps'], event.Jet )
        event['Jet'] = event['Jet'][event['Jet', 'jet_veto_maps']]

    if isRun3:

        if "PNetRegPtRawCorr" in event.Jet.fields:
            event['Jet', 'bRegCorr']       = event.Jet.PNetRegPtRawCorr * event.Jet.PNetRegPtRawCorrNeutrino
        event['Jet', 'btagScore']       = event.Jet.btagPNetB

        ### AGE: Hopefully this is temporarily
        if not isSyntheticData:
            event['Jet'] = ak.where(
                event.Jet.btagScore >= corrections_metadata['btagWP']['L'],
                apply_jerc_corrections(event,
                    corrections_metadata=corrections_metadata,
                    isMC=isMC,
                    run_systematics=False,
                    dataset=dataset,
                    jet_corr_factor=event.Jet.bRegCorr,
                    jet_type="AK4PFPuppiPNetRegressionPlusNeutrino"
                    ),
                apply_jerc_corrections(event,
                    corrections_metadata=corrections_metadata,
                    isMC=isMC,
                    run_systematics=False,
                    dataset=dataset,
                    jet_type="AK4PFPuppi.txt"   ### AGE: .txt is temporary
                    )
            )
            #logging.warning(f"For Run3 we are computing JECs after splitting the jets into btagged and non-btagged jets")

        event['Jet', 'puId']       = 10
        event['Jet', 'pileup'] = ((event.Jet.puId < 7) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose']  = (event.Jet.pt >= 20) & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned & (np.abs(event.Jet.eta) <= 4.7)
        event['Jet', 'selected']        = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned
    else:
        event['Jet', 'calibration'] = event.Jet.pt / ( event.Jet.pt_raw if 'pt_raw' in event.Jet.fields else ak.full_like(event.Jet.pt, 1) )
        event['Jet', 'btagScore']  = event.Jet.btagDeepFlavB
        if ('GluGlu' in dataset): 
            event['Jet', 'corrPuId'] = compute_puid( event.Jet, dataset ) #### To be used in 2024_v2 and above for ALL samples
        else: event['Jet', 'corrPuId'] = ak.where( event.Jet.puId < 7, True, False )
        event['Jet', 'pileup'] = ((event.Jet.corrPuId) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned
        event['Jet', 'selected']      = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned


    if override_selected_with_flavor_bit and "jet_flavor_bit" in event.Jet.fields:
        event['Jet', 'selected'] = (event.Jet.selected) | (event.Jet.jet_flavor_bit == 1)
        event['Jet', 'selected_loose'] = True

    event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)

    event['Jet', 'tagged']       = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])

    event['selJet_no_bRegCorr']  = event.Jet[event.Jet.selected]

    #
    # Apply the bRegCorr to the tagged jets
    #
    bRegCorr_factor_flat = copy(ak.flatten(event.Jet.bRegCorr).to_numpy())
    tagged_flag_flat    = ak.flatten(event.Jet.tagged)
    bRegCorr_factor_flat[~tagged_flag_flat] = 1.0
    bRegCorr_factor = ak.unflatten(bRegCorr_factor_flat, ak.num(event.Jet.bRegCorr) )
    selJet_pvec = event.Jet[event.Jet.selected]  * bRegCorr_factor[event.Jet.selected]
    selJet_pvec["tagged"] = event.Jet[event.Jet.selected].tagged
    selJet_pvec["tagged_loose"] = event.Jet[event.Jet.selected].tagged_loose
    selJet_pvec["btagScore"] = event.Jet[event.Jet.selected].btagScore
    selJet_pvec["puId"] = event.Jet[event.Jet.selected].puId
    selJet_pvec["jetId"] = event.Jet[event.Jet.selected].jetId

    if "hadronFlavour" in event.Jet.fields:
        selJet_pvec["hadronFlavour"] = event.Jet[event.Jet.selected].hadronFlavour

    event['selJet']  = selJet_pvec

    event['passJetMult'] = event.nJet_selected >= 4

    event['nJet_tagged']         = ak.num(event.Jet[event.Jet.tagged])
    event['nJet_tagged_loose']   = ak.num(event.Jet[event.Jet.tagged_loose])

    event['tagJet']              = event.selJet[event.selJet.tagged]
    event['tagJet_loose']        = event.Jet[event.Jet.tagged_loose]

    event['fourTag']  = (event['nJet_tagged'] >= 4)
    event['threeTag'] = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4)
    event['twoTag']   = (event['nJet_tagged_loose'] == 2) & (event['nJet_selected'] >= 4)

    if isRun3:
        event['passPreSel'] = event.twoTag | event.threeTag | event.fourTag
    else:
        event['passPreSel'] = event.threeTag | event.fourTag

    tagCode = np.zeros(len(event), dtype=int)
    tagCode[event.fourTag]  = 4
    tagCode[event.threeTag] = 3
    tagCode[event.twoTag] = 2


    # # For low pt selection
    if run_lowpt_selection:
        event['Jet', 'selected_lowpt'] = (event.Jet.pt >= 15) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned & ~event.Jet.selected
        event['lowptJet'] = event.Jet[event.Jet.selected_lowpt]
        event['Jet', 'tagged_lowpt'] = event.Jet.selected_lowpt & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['M'])
        event['Jet', 'tagged_loose_lowpt'] = event.Jet.selected_lowpt & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['L'])
        event['nJet_tagged_lowpt'] = ak.num(event.Jet[event.Jet.tagged_lowpt])
        event['nJet_tagged_loose_lowpt'] = ak.num(event.Jet[event.Jet.tagged_loose_lowpt])
        event['tagJet_lowpt'] = event.Jet[event.Jet.tagged_lowpt]
        event['lowpt_fourTag']  = (event['nJet_tagged']==3) & (event['nJet_tagged_lowpt'] > 0) & ~event.fourTag
        event['lowpt_threeTag_3b1j_0b'] = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4) & (event['nJet_tagged_loose_lowpt']==0) & (ak.num(event['lowptJet'])>0)
        event['lowpt_threeTag_2b2j_1b'] = (event['nJet_tagged_loose'] == 2) & (event['nJet_selected'] >= 4) & (event['nJet_tagged_loose_lowpt']==1)
        event['lowpt_threeTag_1b3j_2b'] = (event['nJet_tagged_loose'] == 1) & (event['nJet_selected'] >= 4) & (event['nJet_tagged_loose_lowpt']==2)
        event['lowpt_threeTag_0b4j_3b'] = (event['nJet_tagged_loose'] == 0) & (event['nJet_selected'] >= 4) & (event['nJet_tagged_loose_lowpt']==3)
        event['lowpt_threeTag'] = event['lowpt_threeTag_3b1j_0b'] | event['lowpt_threeTag_2b2j_1b'] | event['lowpt_threeTag_1b3j_2b'] | event['lowpt_threeTag_0b4j_3b']
        tagCode[event["lowpt_fourTag"]] = 14
        tagCode[event["lowpt_threeTag"]] = 13
        event['lowpt_categories'] = np.where(
            event['fourTag'], 1,
            np.where(
                event['lowpt_fourTag'], 3,
                np.where(
                    event["lowpt_threeTag_3b1j_0b"], 5,
                    np.where(
                        event["lowpt_threeTag_2b2j_1b"], 7,
                        np.where(
                            event["lowpt_threeTag_1b3j_2b"], 9,
                            np.where(
                                event["lowpt_threeTag_0b4j_3b"], 11,
                                np.where(event['threeTag'], 13, 15)
                            )
                        )
                    )
                )
            )
        )   ### these is the category for the low pt selection

        event['passPreSel'] = event.lowpt_threeTag | event.lowpt_fourTag


    event['tag'] = tagCode

    # Only need 30 GeV jets for signal systematics
    if loosePtForSkim:
        event['Jet', 'selected_lowpt_forskim'] = (event.Jet.pt >= 15) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned
        event['nJet_selected_lowpt_forskim'] = ak.sum(event.Jet.selected_lowpt_forskim, axis=1)
        event['Jet', 'tagged_lowpt_forskim']     = event.Jet.selected_lowpt_forskim & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
        event['passJetMult_lowpt_forskim'] = event.nJet_selected_lowpt_forskim >= 4
        event['nJet_tagged_lowpt_forskim']       = ak.num(event.Jet[event.Jet.tagged_lowpt_forskim])
        event["fourTag_lowpt_forskim"]  = (event['nJet_tagged_lowpt_forskim']     >= 4)
        event['passPreSel_lowpt_forskim'] = event.threeTag | event.fourTag_lowpt_forskim


    #  Calculate hT
    event["hT"] = ak.sum(event.Jet[event.Jet.selected_loose].pt, axis=1)
    event["hT_selected"] = ak.sum(event.Jet[event.Jet.selected].pt, axis=1)
    event["hT_trigger"] = ak.sum(event.Jet[event.Jet.ht_selected].pt, axis=1)


    return event

def apply_object_selection_boosted_4b( event ):
    """docstring for apply_basic_selection_4b. This fuction is not modifying the content of anything in events. it is just adding it"""

    event['FatJet'] = event.FatJet[ ak.argsort( event.FatJet.particleNetMD_Xbb, axis=1, ascending=False )  ]
    event['FatJet', 'selected'] = (event.FatJet.pt > 300) & (np.abs(event.FatJet.eta) < 2.4)
    event['nFatJet_selected'] = ak.sum(event.FatJet.selected, axis=1)

    event['passBoostedKin'] = event.nFatJet_selected >=2
    tmp_selev = event[ event.passBoostedKin ]
    candJet1 = (tmp_selev.FatJet[:,0].msoftdrop > 50) & (tmp_selev.FatJet[:,0].particleNetMD_Xbb > 0.8)
    candJet2 = (tmp_selev.FatJet[:,1].particleNet_mass > 50)
    if 'bdt' in tmp_selev.fields:
        passBDT = (tmp_selev.FatJet[:,1].particleNetMD_Xbb > 0.950) & (tmp_selev.bdt['score']> 0.03)  ### bdt_score only in picoAOD.chunk.withBDT.root files
        # passBDT = (tmp_selev.FatJet[:,1].particleNetMD_Xbb > 0.980) & (tmp_selev.bdt['score']> 0.43)  ### bdt_score only in picoAOD.chunk.withBDT.root files
    else: passBDT = np.full( len(tmp_selev), True )

    passBoostedSel = np.full( len(event), False )
    passBoostedSel[event.passBoostedKin] = (candJet1 & candJet2 & passBDT)
    event['passBoostedSel'] = passBoostedSel


    return event

def create_cand_jet_dijet_quadjet( selev, event_event,
                                   apply_FvT:bool = False,
                                   run_SvB:bool = False,
                                   run_systematics:bool = False,
                                   classifier_SvB = None,
                                   classifier_SvB_MA = None,
                                   processOutput = None,
                                   ):
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
    diJet["dr"] = diJet["lead"].delta_r(diJet["subl"])
    diJet["dphi"] = diJet["lead"].delta_phi(diJet["subl"])

    # Sort diJets within views to be lead st, subl st
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

            ##### AGE: I dont understand why synthetic does not run without this
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

    selev["region"] = ( selev["quadJet_selected"].SR * 0b10 + selev["quadJet_selected"].SB * 0b01 )
    # selev["region"] = ak.zip({
    #     "SR": selev["quadJet_selected"].SR,
    #     "SB": selev["quadJet_selected"].SB
    #     })

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
