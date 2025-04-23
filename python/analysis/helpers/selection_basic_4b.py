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
from coffea.nanoevents.methods.base import NanoEventsArray
from typing import Dict, Any

def muon_selection(muon: ak.Array, isRun3: bool = False) -> ak.Array:
    """
    Selects muons based on kinematic, isolation, and identification criteria.

    Parameters:
    -----------
    muon : ak.Array
        The muon collection containing fields such as `pt`, `eta`, `pfRelIso04_all`, `looseId`, `dz`, and `dxy`.
    isRun3 : bool, optional
        Whether to apply Run 3 selection criteria. Defaults to False.

    Returns:
    --------
    ak.Array
        A boolean mask indicating selected muons.
    """
    muon_kin = (muon.pt > 10) & (abs(muon.eta) < (2.4 if isRun3 else 2.5))
    muon_iso_ID = (muon.pfRelIso04_all < 0.15) & muon.looseId

    if isRun3:
        muon_IP = (
            ((abs(muon.eta) < 1.479) & (abs(muon.dz) < 0.1) & (abs(muon.dxy) < 0.05)) |
            ((abs(muon.eta) >= 1.479) & (abs(muon.dz) < 0.2) & (abs(muon.dxy) < 0.1))
        )
    else:
        muon_IP = True

    return muon_kin & muon_iso_ID & muon_IP


def electron_selection(electron: ak.Array, isRun3: bool = False) -> ak.Array:
    """
    Selects electrons based on kinematic, isolation, and identification criteria.

    Parameters:
    -----------
    electron : ak.Array
        The electron collection containing fields such as `pt`, `eta`, `pfRelIso03_all`, `mvaNoIso_WP90`, `mvaFall17V2Iso_WP90`, `dz`, and `dxy`.
    isRun3 : bool, optional
        Whether to apply Run 3 selection criteria. Defaults to False.

    Returns:
    --------
    ak.Array
        A boolean mask indicating selected electrons.
    """
    electron_kin = (electron.pt > 15) & (abs(electron.eta) < 2.5)
    electron_iso_ID = (electron.pfRelIso03_all < 0.15) & (
        getattr(electron, 'mvaNoIso_WP90') if isRun3 else getattr(electron, 'mvaFall17V2Iso_WP90')
    )

    electron_IP = (
        ((abs(electron.eta) < 1.479) & (abs(electron.dz) < 0.1) & (abs(electron.dxy) < 0.05)) |
        ((abs(electron.eta) >= 1.479) & (abs(electron.dz) < 0.2) & (abs(electron.dxy) < 0.1))
    ) if isRun3 else True

    return electron_kin & electron_iso_ID & electron_IP

def jet_selection(event, corrections_metadata, isRun3: bool = False, isMC: bool = False, isSyntheticData: bool = False, isSyntheticMC: bool = False, dataset: str = ''):
    """
    Applying jet selection, and creating new variables
    """

    if isRun3:

        #if "PNetRegPtRawCorr" in event.Jet.fields:
        #    event['Jet', 'bRegCorr']       = event.Jet.PNetRegPtRawCorr * event.Jet.PNetRegPtRawCorrNeutrino
        event['Jet', 'bRegCorr']       = 1.0
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
                                       jet_corr_factor= event.Jet.PNetRegPtRawCorr * event.Jet.PNetRegPtRawCorrNeutrino,
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
        if ('GluGlu' in dataset) and not isSyntheticMC:
            event['Jet', 'corrPuId'] = compute_puid( event.Jet, dataset ) #### To be used in 2024_v2 and above for ALL samples
        else: event['Jet', 'corrPuId'] = ak.where( event.Jet.puId < 7, True, False )
        event['Jet', 'pileup'] = ((event.Jet.corrPuId) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned
        event['Jet', 'selected']      = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned

    event['Jet', 'tagged']       = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])

    return event


def apply_bRegCorr( jet ):
    '''
    # Apply the bRegCorr to the tagged jets
    '''
    
    bRegCorr_factor_flat = copy(ak.flatten(jet.bRegCorr).to_numpy())
    tagged_flag_flat    = ak.flatten(jet.tagged)
    bRegCorr_factor_flat[~tagged_flag_flat] = 1.0
    bRegCorr_factor = ak.unflatten(bRegCorr_factor_flat, ak.num(jet.bRegCorr) )
    selJet_pvec = jet[jet.selected]  * bRegCorr_factor[jet.selected]
    selJet_pvec["tagged"] = jet[jet.selected].tagged
    selJet_pvec["tagged_loose"] = jet[jet.selected].tagged_loose
    selJet_pvec["btagScore"] = jet[jet.selected].btagScore
    selJet_pvec["puId"] = jet[jet.selected].puId
    selJet_pvec["jetId"] = jet[jet.selected].jetId

    if "hadronFlavour" in jet.fields:
        selJet_pvec["hadronFlavour"] = jet[jet.selected].hadronFlavour

    return selJet_pvec

def lowpt_jet_selection(event, corrections_metadata):

    event['Jet', 'selected_lowpt'] = (event.Jet.pt >= 15) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned & ~event.Jet.selected  ### this can be improved

    event['selJet_lowpt'] = apply_bRegCorr( event.Jet[event.Jet.selected_lowpt] )
    event['nJet_selected_lowpt'] = ak.num(event.selJet_lowpt, axis=1)

    event['Jet', 'tagged_lowpt'] = event.Jet.selected_lowpt & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose_lowpt'] = event.Jet.selected_lowpt & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])
    event['nJet_tagged_lowpt'] = ak.num(event.Jet[event.Jet.tagged_lowpt])
    event['nJet_tagged_loose_lowpt'] = ak.num(event.Jet[event.Jet.tagged_loose_lowpt])
    
    event['tagJet_lowpt'] = event.Jet[event.Jet.tagged_lowpt]

    event['lowpt_fourTag']  = (event['nJet_tagged']==3) & (event['nJet_tagged_lowpt'] > 0) & ~event.fourTag
    event['lowpt_threeTag'] = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4) & (event['nJet_tagged_loose_lowpt']==0) & (event['nJet_selected_lowpt']>0)

    event['tag'] = ak.zip( { 
        "twoTag": event.twoTag,
        "threeTag": event.threeTag,
        "fourTag": event.fourTag,
        "lowpt_fourTag": event.lowpt_fourTag,
        "lowpt_threeTag": event.lowpt_threeTag,
    })

    event['lowpt_categories'] = np.where(
        event['fourTag'], 0,
        np.where(
            event['lowpt_fourTag'], 5,
            np.where(
                event["lowpt_threeTag"], 7,
                    np.where(event['threeTag'], 3, 9)
            )
        )
    )   ### these is the category for the low pt selection

    ## replacing jet variables to not modify the cand jets
    event['passPreSel'] = event.lowpt_threeTag | event.lowpt_fourTag
    event['Jet', 'selected'] = event.Jet.selected | event.Jet.selected_lowpt

    return event


def apply_object_selection_4b(event, corrections_metadata, *,
                                dataset: str = '',
                                doLeptonRemoval: bool = True,
                                loosePtForSkim: bool = False,
                                override_selected_with_flavor_bit: bool = False,
                                do_jet_veto_maps: bool = False,
                                isRun3: bool = False,
                                isMC: bool = False,  ### temporary for Run3
                                isSyntheticData: bool = False,
                                isSyntheticMC: bool = False,
                            ):
    """
    docstring for apply_basic_selection_4b. This fuction is not modifying the content of anything in events. it is just adding it
    """

    #
    # Combined RunII and 3 selection
    #
    event['Muon', 'selected'] = muon_selection(event.Muon, isRun3)
    # event['nMuon_selected'] = ak.sum(event.Muon.selected, axis=1)
    event['selMuon'] = event.Muon[event.Muon.selected]

    if 'Electron' in event.fields:
        event['Electron', 'selected'] = electron_selection(event.Electron, isRun3)   
        # event['nElectron_selected'] = ak.sum(event.Electron.selected, axis=1)
        event['selElec'] = event.Electron[event.Electron.selected]
        selLepton = ak.concatenate( [event.selElec, event.selMuon], axis=1 )
    else: selLepton = event.selMuon

    event['Jet', 'lepton_cleaned'] = np.full(len(event), True) if not doLeptonRemoval else drClean( event.Jet, selLepton )[1]  ### 0 is the collection of jets, 1 is the flag

    if do_jet_veto_maps:
        event['Jet', 'jet_veto_maps'] = apply_jet_veto_maps( corrections_metadata['jet_veto_maps'], event.Jet )
        event['Jet'] = event['Jet'][event['Jet', 'jet_veto_maps']]

    event = jet_selection(event, corrections_metadata, isRun3, isMC, isSyntheticData, isSyntheticMC, dataset)

    if override_selected_with_flavor_bit and "jet_flavor_bit" in event.Jet.fields:
        event['Jet', 'selected'] = (event.Jet.selected) | (event.Jet.jet_flavor_bit == 1)
        event['Jet', 'selected_loose'] = True

    event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)
    event['passJetMult'] = event['nJet_selected'] >= 4

    event['selJet_no_bRegCorr']  = event.Jet[event.Jet.selected]
    event['selJet'] = apply_bRegCorr(event.Jet)

    event['tagJet']              = event.selJet[event.selJet.tagged]
    event['tagJet_loose']        = event.selJet[event.selJet.tagged_loose]

    event['nJet_tagged']         = ak.num(event.tagJet)
    event['nJet_tagged_loose']   = ak.num(event.tagJet_loose)

    event['fourTag']  = (event['nJet_tagged'] >= 4)
    event['threeTag'] = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4)
    event['twoTag']   = (event['nJet_tagged_loose'] == 2) & (event['nJet_selected'] >= 4)

    if isSyntheticData or isSyntheticMC:
        event['threeTag'] = False
        event['twoTag']   = False

    if isRun3:
        event['passPreSel'] = event.twoTag | event.threeTag | event.fourTag
    else:
        event['passPreSel'] = event.threeTag | event.fourTag

    event['tag'] = ak.zip( { 
        "twoTag": event.twoTag,
        "threeTag": event.threeTag,
        "fourTag": event.fourTag,
    })

    # For trigger emulation
    event['Jet', 'muon_cleaned'] = drClean( event.Jet, event.selMuon )[1]
    event['Jet', 'ht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) & event.Jet.muon_cleaned
    #  Calculate hT
    event["hT"] = ak.sum(event.Jet[event.Jet.selected_loose].pt, axis=1)
    event["hT_selected"] = ak.sum(event.Jet[event.Jet.selected].pt, axis=1)
    event["hT_trigger"] = ak.sum(event.Jet[event.Jet.ht_selected].pt, axis=1)

    # Only need 30 GeV jets for signal systematics
    if loosePtForSkim:
        event['Jet', 'selected_lowpt_forskim'] = (event.Jet.pt >= 15) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned
        event['nJet_selected_lowpt_forskim'] = ak.sum(event.Jet.selected_lowpt_forskim, axis=1)
        event['Jet', 'tagged_lowpt_forskim']     = event.Jet.selected_lowpt_forskim & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
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
        # passBDT = (tmp_selev.FatJet[:,1].particleNetMD_Xbb > 0.980) & (tmp_selev.bdt['score']> 0.43)  ### bdt_score only in picoAOD.chunk.withBDT.root files
    else: passBDT = np.full( len(tmp_selev), True )

    passBoostedSel = np.full( len(event), False )
    passBoostedSel[event.passBoostedKin] = (candJet1 & candJet2 & passBDT)
    event['passBoostedSel'] = passBoostedSel

    return event


def apply_dilep_ttbar_selection(event: ak.Array, isRun3: bool = False) -> ak.Array:
    """
    Applies dilepton ttbar selection criteria to the event data.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Muon`, `Electron`, and `MET`.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.

    Returns:
    --------
    ak.Array
        A boolean mask indicating events passing the dilepton ttbar selection criteria.
    """
    # Select muons and electrons
    muon_mask = muon_selection(event.Muon, isRun3)
    electron_mask = electron_selection(event.Electron, isRun3) if 'Electron' in event.fields else None

    # Count selected leptons
    n_muons = ak.sum(muon_mask, axis=1)
    n_electrons = ak.sum(electron_mask, axis=1) if electron_mask is not None else 0

    # Require exactly two leptons (muons + electrons)
    dilepton_mask = (n_muons + n_electrons) == 2

    # Require opposite-sign leptons
    if ('charge' in event.Muon.fields) and ( 'charge' in event.Electron.fields):
        muons = event.Muon[muon_mask]
        electrons = event.Electron[electron_mask] if electron_mask is not None else None
        os_muons = ak.any(muons.charge[:, None] + muons.charge == 0, axis=1)
        os_electrons = ak.any(electrons.charge[:, None] + electrons.charge == 0, axis=1) if electrons is not None else False
        os_muon_electron = ak.any(muons.charge[:, None] + electrons.charge == 0, axis=1) if electrons is not None else False
        opposite_sign_mask = os_muons | os_electrons | os_muon_electron
    else:
        opposite_sign_mask = np.full(len(event), True)

    # Require MET > 40 GeV
    met_mask = event.MET.pt > 40

    # Combine all selection criteria
    selection_mask = dilepton_mask & opposite_sign_mask & met_mask

    return selection_mask
