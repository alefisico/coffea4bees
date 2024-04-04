import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from analysis.helpers.common import init_jet_factory, jet_corrections, mask_event_decision, drClean
from coffea.lumi_tools import LumiMask

def apply_event_selection_4b( event, isMC, corrections_metadata, *, isMixedData=False):

    lumimask = LumiMask(corrections_metadata['goldenJSON'])
    event['lumimask'] = np.full(len(event), True) if isMC else np.array( lumimask(event.run, event.luminosityBlock) )

    event['passHLT'] = np.full(len(event), True) if (isMC or isMixedData) else mask_event_decision( event, decision="OR", branch="HLT", list_to_mask=event.metadata['trigger']  )

    event['passNoiseFilter'] = np.full(len(event), True) if isMixedData else mask_event_decision( event, decision="AND", branch="Flag",
                                                                                                  list_to_mask=corrections_metadata['NoiseFilter'],
                                                                                                  list_to_skip=['BadPFMuonDzFilter', 'hfNoisyHitsFilter']  )

    return event

def apply_object_selection_4b( event, year, isMC, dataset, corrections_metadata, *, isMixedData=False  ):
    """docstring for apply_basic_selection_4b. This fuction is not modifying the content of anything in events. it is just adding it"""

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
        event['Electron', 'selected'] = (event.Electron.pt > 15) & (abs(event.Electron.eta) < 2.5) & (event.Electron.pfRelIso03_all < 0.15) & getattr(event.Electron, ('mvaFall17V2Iso_WP90' if year.startswith('UL') else 'mvaIso_WP90' ) )
        event['nElectron_selected'] = ak.sum(event.Electron.selected, axis=1)
        event['selElec'] = event.Electron[event.Electron.selected]
        selLepton = ak.concatenate( [event.selElec, event.selMuon], axis=1 )
    else: selLepton = event.selMuon

    #
    # Calculate and apply Jet Energy Calibration
    #
#    juncWS = [ corrections_metadata["JERC"][0].replace('STEP', istep)
#               for istep in ['L1FastJet', 'L2Relative', 'L2L3Residual', 'L3Absolute'] ]      ###### AGE: to be reviewed for data, but should be remove with jsonpog
#    if isMC:
#        juncWS += corrections_metadata["JERC"][1:]

    if isMixedData:

        event['Jet', 'pileup'] = ((event.Jet.puId < 0b110) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & ~event.Jet.pileup & (event.Jet.jetId>=2)
        event['Jet', 'selected'] = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2)

    else:
        #jet_variations = init_jet_factory(juncWS, event, isMC)  #### currently creates the pt_raw branch
        #jet_tmp = jet_corrections( event.Jet, event.fixedGridRhoFastjetAll, jec_type=['L1L2L3Res'])   # AGE: jsonpog+correctionlib but not final, that is why it is not used yet

        event['Jet', 'calibration'] = event.Jet.pt / ( event.Jet.pt_raw if 'pt_raw' in event.Jet.fields else ak.full_like(event.Jet.pt, 1) )
        event['Jet', 'lepton_cleaned'] = drClean( event.Jet, selLepton )[1]  ### 0 is the collection of jets, 1 is the flag

        event['Jet', 'pileup'] = ((event.Jet.puId < 0b110) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned
        event['Jet', 'selected'] = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & (event.Jet.jetId>=2) & event.Jet.lepton_cleaned


    event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)
    event['selJet'] = event.Jet[event.Jet.selected]
    event['Jet', 'tagged']       = event.Jet.selected & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['M'])
    event['Jet', 'tagged_loose'] = event.Jet.selected & (event.Jet.btagDeepFlavB >= corrections_metadata['btagWP']['L'])

    event['passJetMult'] = event.nJet_selected >= 4

    event['nJet_tagged']         = ak.num(event.Jet[event.Jet.tagged])
    event['nJet_tagged_loose']   = ak.num(event.Jet[event.Jet.tagged_loose])
    event['tagJet']              = event.Jet[event.Jet.tagged]
    event['tagJet_loose']        = event.Jet[event.Jet.tagged_loose]

    fourTag  = (event['nJet_tagged']       >= 4)
    threeTag = (event['nJet_tagged_loose'] == 3) & (event['nJet_selected'] >= 4)

    event[ 'fourTag']   =  fourTag
    event['threeTag']   = threeTag

    event['passPreSel'] = event.threeTag | event.fourTag

    tagCode = np.full(len(event), 0, dtype=int)
    tagCode[event.fourTag]  = 4
    tagCode[event.threeTag] = 3
    event['tag'] = tagCode

    return event
