import numpy as np
import awkward as ak
from coffea.lumi_tools import LumiMask
from analysis.helpers.common import mask_event_decision
from analysis.helpers.selection_basic_4b import (
    muon_selection,
    electron_selection,
    jet_selection
)

def apply_event_selection(
    event: ak.Array, 
    corrections_metadata: dict, 
    cut_on_lumimask: bool = True
) -> ak.Array:
    """
    Applies basic event selection criteria for a 4b analysis.

    Parameters:
    -----------
    event : awkward.Array
        The event data containing fields such as `run`, `luminosityBlock`, `HLT`, and `Flag`.
    corrections_metadata : dict
        Metadata containing corrections and configuration information, such as:
        - `goldenJSON`: Path to the golden JSON file for luminosity masking.
        - `NoiseFilter`: List of noise filters to apply.
    cut_on_lumimask : bool, optional
        Whether to apply the luminosity mask cut. Defaults to True.

    Returns:
    --------
    event : awkward.Array
        The input event data with additional fields:
        - `lumimask`: Boolean mask indicating events passing the luminosity mask.
        - `passHLT`: Boolean mask indicating events passing the HLT trigger.
        - `passNoiseFilter`: Boolean mask indicating events passing noise filters.
    """
    # Apply luminosity mask
    if 'goldenJSON' not in corrections_metadata:
        raise KeyError("Missing 'goldenJSON' in corrections_metadata.")
    lumimask = LumiMask(corrections_metadata['goldenJSON'])
    event['lumimask'] = (
        np.array(lumimask(event.run, event.luminosityBlock))
        if cut_on_lumimask else np.full(len(event), True)
    )

    # Apply HLT trigger mask
    event['passHLT'] = (
        np.full(len(event), True)
        if 'HLT' not in event.fields else mask_event_decision(
            event, decision="OR", branch="HLT", list_to_mask=event.metadata.get('trigger', [])
        )
    )

    # Apply noise filter mask
    noise_filters = corrections_metadata.get('NoiseFilter', [])
    event['passNoiseFilter'] = (
        np.full(len(event), True)
        if 'Flag' not in event.fields else mask_event_decision(
            event, decision="AND", branch="Flag", list_to_mask=noise_filters,
            list_to_skip=['BadPFMuonDzFilter', 'hfNoisyHitsFilter']
        )
    )

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
    muons = event.selMuon
    n_muons = ak.sum(muons.pt, axis=1)

    electrons = event.selElec 
    if 'selElec' in event.fields:
        electrons = event.selElec
        n_electrons = ak.sum(electrons.pt, axis=1) 
        # Require exactly two leptons (muons + electrons)
        dilepton_mask = (n_muons + n_electrons) == 2
    else:
        dilepton_mask = n_muons == 2

    # Require opposite-sign leptons
    if hasattr(muons, 'charge'):
        os_muons = ak.any(muons.charge[:, None] + muons.charge == 0, axis=1)
        opposite_sign_mask = os_muons         
    elif hasattr(electrons, 'charge'):
        os_electrons = ak.any(electrons.charge[:, None] + electrons.charge == 0, axis=1)
        os_muon_electron = ak.any(muons.charge[:, None] + electrons.charge == 0, axis=1)
        opposite_sign_mask = os_muons | os_electrons | os_muon_electron
    else:
        opposite_sign_mask = np.full(len(event), False)

    # Require MET > 40 GeV
    met_mask = event.MET.pt > 40

    # Combine all selection criteria
    selection_mask = dilepton_mask & opposite_sign_mask & met_mask

    return selection_mask

def apply_boosted_4b_selection(event: ak.Array) -> ak.Array:
    """
    Applies boosted object selection criteria for 4b analysis.

    Parameters:
    -----------
    event : ak.Array
        The event data containing fields such as `FatJet` and `bdt`.

    Returns:
    --------
    ak.Array
        The input event data with additional fields:
        - `FatJet['selected']`: Boolean mask for selected boosted jets.
        - `nFatJet_selected`: Number of selected boosted jets.
        - `passBoostedKin`: Boolean mask for events passing boosted kinematic selection.
        - `passBoostedSel`: Boolean mask for events passing boosted selection criteria.
    """
    # Sort FatJets by particleNetMD_Xbb in descending order
    event['FatJet'] = event.FatJet[ak.argsort(event.FatJet.particleNetMD_Xbb, axis=1, ascending=False)]

    # Apply kinematic selection to FatJets
    event['FatJet', 'selected'] = (event.FatJet.pt > 300) & (np.abs(event.FatJet.eta) < 2.4)
    event['nFatJet_selected'] = ak.sum(event.FatJet.selected, axis=1)

    # Check if events pass boosted kinematic selection
    event['passBoostedKin'] = event.nFatJet_selected >= 2

    # Apply additional selection criteria to candidate jets
    tmp_selev = event[event.passBoostedKin]
    candJet1 = (tmp_selev.FatJet[:, 0].msoftdrop > 50) & (tmp_selev.FatJet[:, 0].particleNetMD_Xbb > 0.8)
    candJet2 = (tmp_selev.FatJet[:, 1].particleNet_mass > 50)

    # Apply BDT-based selection if available
    if 'bdt' in tmp_selev.fields:
        passBDT = (tmp_selev.FatJet[:, 1].particleNetMD_Xbb > 0.950) & (tmp_selev.bdt['score'] > 0.03)  ### bdt_score only in picoAOD.chunk.withBDT.root files
        # passBDT = (tmp_selev.FatJet[:, 1].particleNetMD_Xbb > 0.980) & (tmp_selev.bdt['score'] > 0.43)  ### bdt_score only in picoAOD.chunk.withBDT.root files
    else:
        passBDT = np.full(len(tmp_selev), True)

    # Combine all selection criteria
    passBoostedSel = np.full(len(event), False)
    passBoostedSel[event.passBoostedKin] = candJet1 & candJet2 & passBDT
    event['passBoostedSel'] = passBoostedSel

    return event