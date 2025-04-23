import logging

import numpy as np
import yaml
from analysis.helpers.common import apply_jerc_corrections
from analysis.helpers.mc_weight_outliers import OutlierByMedian
from analysis.helpers.processor_config import processor_config
from analysis.helpers.selection_basic_4b import (
    apply_object_selection_4b,
)
from analysis.helpers.event_selection import apply_event_selection
from coffea.analysis_tools import PackedSelection, Weights
from skimmer.processor.picoaod import PicoAOD


class Skimmer(PicoAOD):
    def __init__(self, loosePtForSkim=False, skim4b=False, mc_outlier_threshold:int|None=200, *args, **kwargs):
        if skim4b:
            kwargs["pico_base_name"] = f'picoAOD_fourTag'
        super().__init__(*args, **kwargs)
        self.loosePtForSkim = loosePtForSkim
        self.skim4b = skim4b
        self.corrections_metadata = yaml.safe_load(open('analysis/metadata/corrections.yml', 'r'))
        self.mc_outlier_threshold = mc_outlier_threshold



    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']

        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'config={config}\n')

        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"] )

        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections(event,
                                      corrections_metadata=self.corrections_metadata[year],
                                      isMC=config["isMC"],
                                      run_systematics=False,
                                      dataset=dataset
                                      )
            event["Jet"] = jets

        event = apply_object_selection_4b( event, self.corrections_metadata[year],
            dataset=dataset,
            doLeptonRemoval=config["do_lepton_jet_cleaning"],
            loosePtForSkim=self.loosePtForSkim,
            isRun3=config["isRun3"],
            isMC=config["isMC"],
            )

        weights = Weights(len(event), storeIndividual=True)

        #
        # general event weights
        #
        if config["isMC"]:
            weights.add( "genweight_", event.genWeight )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        
        if self.loosePtForSkim:
            selections.add( 'passJetMult_lowpt_forskim', event.passJetMult_lowpt_forskim )
            selections.add( "passPreSel_lowpt_forskim",  event.passPreSel_lowpt_forskim)
            final_selection = selections.require( lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult_lowpt_forskim=True, passPreSel_lowpt_forskim=True )
        elif self.skim4b:
            selections.add( 'passJetMult',   event.passJetMult )
            selections.add( "passPreSel",    event.passPreSel)
            selections.add( "passFourTag",    event.fourTag)
            final_selection = selections.require( lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True, passFourTag=True )
        else:
            selections.add( 'passJetMult',   event.passJetMult )
            selections.add( "passPreSel",    event.passPreSel)
            final_selection = selections.require( lumimask=True, passNoiseFilter=True, passHLT=True, passJetMult=True, passPreSel=True )
    
        event["weight"] = weights.weight()

        self._cutFlow.fill( "all",             event, allTag=True )
        cumulative_cuts = []
        for cut in selections.names:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        # debug_mask = ((event.event == 110614) & (event.run == 275890) & (event.luminosityBlock == 1))
        # debug_event = event[debug_mask]
        # print(f"debug {debug_event.fourTag} {debug_event.threeTag} {debug_event.nJet_tagged} {debug_event.nJet_tagged_loose} {debug_event.nJet_selected} {debug_event.Jet.tagged} {debug_event.Jet.selected} {debug_event.Jet.btagScore}")
        # print(f"debug {debug_event.passHLT} {debug_event.passJetMult} {debug_event.passPreSel} {debug_event.Jet.pt} {debug_event.Jet.pt_raw} \n\n\n")

        return final_selection

    def preselect(self, event):
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        config = processor_config(processName, dataset, event)
        if config["isMC"] and self.mc_outlier_threshold is not None and "genWeight" in event.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(event.genWeight)
