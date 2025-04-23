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
        A collection of selected muons.
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

    muon['selected'] = muon_kin & muon_iso_ID & muon_IP

    return muon[muon.selected]


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

    electron['selected'] = electron_kin & electron_iso_ID & electron_IP

    return electron[electron.selected]

def lepton_selection(event: ak.Array, isRun3: bool = False) -> ak.Array:
    """
    Selects leptons (muons and electrons) and adds them to the event.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Muon` and `Electron`.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.

    Returns:
    --------
    ak.Array
        The input event data with additional fields:
        - `selMuon`: Selected muons.
        - `selElec`: Selected electrons (if present).
    """
    # Select muons
    event['selMuon'] = muon_selection(event.Muon, isRun3)

    # Select electrons if present
    if 'Electron' in event.fields:
        event['selElec'] = electron_selection(event.Electron, isRun3)
        event['selLepton'] = ak.concatenate([event.selElec, event.selMuon], axis=1)
    else:
        event['selLepton'] = event.selMuon

    return event

def jet_selection(event: ak.Array, corrections_metadata: dict, isRun3: bool = False, isMC: bool = False, isSyntheticData: bool = False, isSyntheticMC: bool = False, dataset: str = '', doLeptonRemoval: bool = True, do_jet_veto_maps: bool = False, override_selected_with_flavor_bit: bool = False) -> ak.Array:
    """
    Applies jet selection criteria and creates new variables for the event data.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `Jet`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information, such as b-tagging working points.
    isRun3 : bool, optional
        Whether to apply Run 3-specific selection criteria. Defaults to False.
    isMC : bool, optional
        Whether the data is Monte Carlo simulation. Defaults to False.
    isSyntheticData : bool, optional
        Whether the data is synthetic. Defaults to False.
    isSyntheticMC : bool, optional
        Whether the Monte Carlo data is synthetic. Defaults to False.
    dataset : str, optional
        The dataset name. Defaults to an empty string.
    doLeptonRemoval : bool, optional
        Whether to perform lepton removal. Defaults to True.
    do_jet_veto_maps : bool, optional
        Whether to apply jet veto maps. Defaults to False.
    override_selected_with_flavor_bit : bool, optional
        Whether to override selected jets with flavor bit. Defaults to False.

    Returns:
    --------
    ak.Array
        The input event data with additional fields for jet selection and tagging:
        - `Jet['lepton_cleaned']`: Boolean mask for jets cleaned of leptons.
        - `Jet['jet_veto_maps']`: Boolean mask for jets passing veto maps (if applied).
        - `Jet['bRegCorr']`: Regression correction factor for jets (Run3 only).
        - `Jet['btagScore']`: B-tagging score for jets.
        - `Jet['pileup']`: Boolean mask for pileup jets.
        - `Jet['selected_loose']`: Boolean mask for loosely selected jets.
        - `Jet['selected']`: Boolean mask for selected jets.
        - `Jet['tagged']`: Boolean mask for b-tagged jets.
        - `Jet['tagged_loose']`: Boolean mask for loosely b-tagged jets.
        - `nJet_selected`: Number of selected jets.
        - `selJet_no_bRegCorr`: Jets selected without bRegCorr applied.
        - `selJet`: Jets selected with bRegCorr applied.
        - `tagJet`: Jets tagged as b-jets.
        - `tagJet_loose`: Jets loosely tagged as b-jets.
        - `nJet_tagged`: Number of b-tagged jets.
        - `nJet_tagged_loose`: Number of loosely b-tagged jets.
    """
    # Initialize lepton-cleaned jets
    event['Jet', 'lepton_cleaned'] = np.full(len(event), True) if not doLeptonRemoval else drClean(event.Jet, event['selLepton'])[1]

    # Apply jet veto maps if required
    if do_jet_veto_maps:
        event['Jet', 'jet_veto_maps'] = apply_jet_veto_maps(corrections_metadata['jet_veto_maps'], event.Jet)
        event['Jet'] = event['Jet'][event['Jet', 'jet_veto_maps']]

    # Run3-specific jet selection
    if isRun3:
        event['Jet', 'bRegCorr'] = 1.0
        event['Jet', 'btagScore'] = event.Jet.btagPNetB

        if not isSyntheticData:
            event['Jet'] = ak.where(
                event.Jet.btagScore >= corrections_metadata['btagWP']['L'],
                apply_jerc_corrections(
                    event,
                    corrections_metadata=corrections_metadata,
                    isMC=isMC,
                    run_systematics=False,
                    dataset=dataset,
                    jet_corr_factor=event.Jet.PNetRegPtRawCorr * event.Jet.PNetRegPtRawCorrNeutrino,
                    jet_type="AK4PFPuppiPNetRegressionPlusNeutrino"
                ),
                apply_jerc_corrections(
                    event,
                    corrections_metadata=corrections_metadata,
                    isMC=isMC,
                    run_systematics=False,
                    dataset=dataset,
                    jet_type="AK4PFPuppi.txt"
                )
            )

        event['Jet', 'pileup'] = ((event.Jet.puId < 7) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & (event.Jet.jetId >= 2) & event.Jet.lepton_cleaned & (np.abs(event.Jet.eta) <= 4.7)
        event['Jet', 'selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId >= 2) & event.Jet.lepton_cleaned

    # Non-Run3 jet selection
    else:
        event['Jet', 'calibration'] = event.Jet.pt / (event.Jet.pt_raw if 'pt_raw' in event.Jet.fields else ak.full_like(event.Jet.pt, 1))
        event['Jet', 'btagScore'] = event.Jet.btagDeepFlavB

        if ('GluGlu' in dataset) and not isSyntheticMC:
            event['Jet', 'corrPuId'] = compute_puid(event.Jet, dataset)
        else:
            event['Jet', 'corrPuId'] = ak.where(event.Jet.puId < 7, True, False)

        event['Jet', 'pileup'] = ((event.Jet.corrPuId) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & ~event.Jet.pileup & (event.Jet.jetId >= 2) & event.Jet.lepton_cleaned
        event['Jet', 'selected'] = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId >= 2) & event.Jet.lepton_cleaned

    # Tagging jets
    event['Jet', 'tagged'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagScore >= corrections_metadata['btagWP']['L'])

    # Override selected jets with flavor bit if required
    if override_selected_with_flavor_bit and "jet_flavor_bit" in event.Jet.fields:
        event['Jet', 'selected'] = (event.Jet.selected) | (event.Jet.jet_flavor_bit == 1)
        event['Jet', 'selected_loose'] = True

    # Count selected jets
    event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)

    # Additional variables
    event['selJet_no_bRegCorr'] = event.Jet[event.Jet.selected]
    event['selJet'] = apply_bRegCorr(event.Jet)
    event['tagJet'] = event.selJet[event.selJet.tagged]
    event['tagJet_loose'] = event.selJet[event.selJet.tagged_loose]
    event['nJet_tagged'] = ak.num(event.tagJet)
    event['nJet_tagged_loose'] = ak.num(event.tagJet_loose)

    # For trigger emulation
    event['Jet', 'muon_cleaned'] = drClean(event.Jet, event.selMuon)[1]
    event['Jet', 'ht_selected'] = (event.Jet.pt >= 30) & (np.abs(event.Jet.eta) < 2.4) & event.Jet.muon_cleaned

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
