import numpy as np
import awkward as ak
from analysis.helpers.common import mask_event_decision, drClean
from analysis.helpers.SvB_helpers import compute_SvB
from coffea.lumi_tools import LumiMask
from base_class.math.random import Squares
from copy import copy

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
                              run_lowpt_selection: bool = False):
    """docstring for apply_basic_selection_4b. This fuction is not modifying the content of anything in events. it is just adding it"""

    ## Very simplified selection for Run 3
    if '202' in dataset:  ### Run 3 is only with 202X years
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 10) & (np.abs(event.Jet.eta) <= 4.7)
        event['Jet', 'selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) <= 2.4) & (event.Jet.jetId>=2)
        event['Jet', 'tagged']       = event.Jet.selected & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['M'])
        # event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['L'])
        event['selJet'] = event.Jet[event.Jet.selected]
        event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)

        if '2022' in dataset:
            event['passJetMult'] = ak.where( event.nJet_selected >= 4, True, False)
            #event['selJet'] = ak.pad_none(event.selJet, 4, axis=1)
            #event['passJetMult'] = ak.where(
            #    (event.nJet_selected >= 4) &
            #    (event.selJet[:, 0].pt > 70) &
            #    (event.selJet[:, 1].pt > 50) &
            #    (event.selJet[:, 2].pt > 40) &
            #    (event.selJet[:, 3].pt > 35),
            #    True,
            #    False
            #    )
        else:
            event['passJetMult'] = ak.where( event.nJet_selected >= 4, True, False)

        event['nJet_tagged']         = ak.num(event.Jet[event.Jet.tagged])
        # event['nJet_tagged_loose']   = ak.num(event.Jet[event.Jet.tagged_loose])
        event['tagJet']              = event.selJet[event.selJet.tagged]
        # event['tagJet_loose']        = event.Jet[event.Jet.tagged_loose]

        event['fourTag']  = (event['nJet_tagged'] >= 4)
        event['threeTag']   = (event['nJet_tagged'] == 3) & (event['nJet_selected'] >= 4)
        event['twoTag']   = (event['nJet_tagged'] == 2) & (event['nJet_selected'] >= 4)

        event['passPreSel'] = event.twoTag | event.threeTag | event.fourTag

        ## for skims
        event['passJetMult_lowpt_forskim'] = event.passJetMult
        event['passPreSel_lowpt_forskim'] = event.passPreSel

    ## For Run 2
    else:
        #
        # Adding muons (loose muon id definition)
        # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        #
        event['Muon', 'selected'] = (event.Muon.pt > 10) & (abs(event.Muon.eta) < 2.5) & (event.Muon.pfRelIso04_all < 0.15) & (event.Muon.looseId)
        event['nMuon_selected'] = ak.sum(event.Muon.selected, axis=1)
        event['selMuon'] = event.Muon[event.Muon.selected]

        #
        # Adding electrons (loose electron id)
        # https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
        #
        if 'Electron' in event.fields:
            event['Electron', 'selected'] = (event.Electron.pt > 15) & (abs(event.Electron.eta) < 2.5) & (event.Electron.pfRelIso03_all < 0.15) & getattr(event.Electron, 'mvaFall17V2Iso_WP90' )
            event['nElectron_selected'] = ak.sum(event.Electron.selected, axis=1)
            event['selElec'] = event.Electron[event.Electron.selected]
            selLepton = ak.concatenate( [event.selElec, event.selMuon], axis=1 )
        else: selLepton = event.selMuon


        event['Jet', 'calibration'] = event.Jet.pt / ( event.Jet.pt_raw if 'pt_raw' in event.Jet.fields else ak.full_like(event.Jet.pt, 1) )

        if doLeptonRemoval:
            event['Jet', 'lepton_cleaned'] = drClean( event.Jet, selLepton )[1]  ### 0 is the collection of jets, 1 is the flag
        else:
            event['Jet', 'lepton_cleaned'] = np.full(len(event), True)

        # For trigger emulation
        event['Jet', 'muon_cleaned'] = drClean( event.Jet, event.selMuon )[1]  ### 0 is the collection of jets, 1 is the flag
        event['Jet', 'ht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) & event.Jet.muon_cleaned

        event['Jet', 'pileup'] = ((event.Jet.puId < 7) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned
        event['Jet', 'selected'] = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned

        if override_selected_with_flavor_bit and "jet_flavor_bit" in event.Jet.fields:
            event['Jet', 'selected'] = (event.Jet.selected) | (event.Jet.jet_flavor_bit == 1)
            event['Jet', 'selected_loose'] = True


        event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)

        event['Jet', 'tagged']       = event.Jet.selected & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['M'])
        event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['L'])
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
        selJet_pvec["btagDeepFlavB"] = event.Jet[event.Jet.selected].btagDeepFlavB
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

        event['fourTag']    = (event['nJet_tagged']       >= 4)
        event['threeTag']   = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4)

        event['passPreSel'] = event.threeTag | event.fourTag

        tagCode = np.zeros(len(event), dtype=int)
        tagCode[event.fourTag]  = 4
        tagCode[event.threeTag] = 3

        #  Calculate hT
        event["hT"] = ak.sum(event.Jet[event.Jet.selected_loose].pt, axis=1)
        event["hT_selected"] = ak.sum(event.Jet[event.Jet.selected].pt, axis=1)
        event["hT_trigger"] = ak.sum(event.Jet[event.Jet.ht_selected].pt, axis=1)

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
            event['Jet', 'tagged_lowpt_forskim']     = event.Jet.selected_lowpt_forskim & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['M'])
            event['passJetMult_lowpt_forskim'] = event.nJet_selected_lowpt_forskim >= 4
            event['nJet_tagged_lowpt_forskim']       = ak.num(event.Jet[event.Jet.tagged_lowpt_forskim])
            event["fourTag_lowpt_forskim"]  = (event['nJet_tagged_lowpt_forskim']     >= 4)
            event['passPreSel_lowpt_forskim'] = event.threeTag | event.fourTag_lowpt_forskim

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
    else: passBDT = np.full( len(tmp_selev), True )

    passBoostedSel = np.full( len(event), False )
    passBoostedSel[event.passBoostedKin] = (candJet1 & candJet2 & passBDT)
    event['passBoostedSel'] = passBoostedSel


    return event

def create_cand_jet_dijet_quadjet( selev, event_event,
                                   isMC:bool = False,
                                   apply_FvT:bool = False,
                                   apply_boosted_veto:bool = False,
                                   run_SvB:bool = False,
                                   run_systematics:bool = False,
                                   classifier_SvB = None,
                                   classifier_SvB_MA = None,
                                   ):
    #
    # Build and select boson candidate jets with bRegCorr applied
    #
    sorted_idx = ak.argsort( selev.Jet.btagDeepFlavB * selev.Jet.selected, axis=1, ascending=False )
    canJet_idx = sorted_idx[:, 0:4]
    notCanJet_idx = sorted_idx[:, 4:]

    # # apply bJES to canJets
    canJet = selev.Jet[canJet_idx] * selev.Jet[canJet_idx].bRegCorr
    canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
    canJet["btagDeepFlavB"] = selev.Jet.btagDeepFlavB[canJet_idx]
    canJet["puId"] = selev.Jet.puId[canJet_idx]
    canJet["jetId"] = selev.Jet.jetId[canJet_idx]

    if "hadronFlavour" in selev.Jet.fields:
        canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]
    canJet["calibration"] = selev.Jet.calibration[canJet_idx]

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
    rng = Squares("quadJetSelection", event_event)
    counter = np.zeros((len(selev), 3, 2), dtype=np.uint64)
    counter[:, :, 0] = np.round(np.asarray(diJet[:, :, 0].mass), 0).view(np.uint64)
    counter[:, :, 1] = np.round(np.asarray(diJet[:, :, 1].mass), 0).view(np.uint64)

    # print(f"{self.chunk} mass {diJet[:, :, 0].mass[0:5]}\n")
    # print(f"{self.chunk} mass view64 {np.asarray(diJet[:, :, 0].mass).view(np.uint64)[0:5]}\n")
    # print(f"{self.chunk} mass rounded view64 {np.round(np.asarray(diJet[:, :, 0].mass), 0).view(np.uint64)[0:5]}\n")
    # print(f"{self.chunk} mass rounded {np.round(np.asarray(diJet[:, :, 0].mass), 0)[0:5]}\n")
    # print(f"{self.chunk} counter 0 {counter[:, :, 0][0:5]}\n")
    # print(f"{self.chunk} counter 1 {counter[:, :, 1][0:5]}\n")

    quadJet = ak.zip( { "lead": diJet[:, :, 0],
                        "subl": diJet[:, :, 1],
                        "close": diJetDr[:, :, 0],
                        "other": diJetDr[:, :, 1],
                        "passDiJetMass": ak.all(diJet.passDiJetMass, axis=2),
                        "random": rng.uniform(counter, low=0.1, high=0.9),
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
    quadJet["HHSR"] = ((quadJet.xHH < max_xHH) & selev.vetoBoostedSel ) if apply_boosted_veto else (quadJet.xHH < max_xHH)
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
                        classifier_SvB,
                        classifier_SvB_MA,
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

    if run_SvB:
        selev["passSvB"] = selev["SvB_MA"].ps > 0.80
        selev["failSvB"] = selev["SvB_MA"].ps < 0.05
