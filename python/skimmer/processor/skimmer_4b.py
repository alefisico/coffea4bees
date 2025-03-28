import logging

import numpy as np
import yaml
from analysis.helpers.common import apply_jerc_corrections
from analysis.helpers.mc_weight_outliers import OutlierByMedian
from analysis.helpers.processor_config import processor_config
from analysis.helpers.selection_basic_4b import (
    apply_event_selection_4b,
    apply_object_selection_4b,
)
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

        event = apply_event_selection_4b( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"] )

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
        selections.add( 'passJetMult',   event.passJetMult )
        if self.loosePtForSkim:
            selections.add( "passPreSel_lowpt_forskim",  event.passPreSel_lowpt_forskim)
        selections.add( "passPreSel",    event.passPreSel)
        if self.skim4b:
            selections.add( "passFourTag",    event.fourTag)

        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        if self.loosePtForSkim:
            all_cuts = ["passNoiseFilter", "passHLT", "passJetMult_lowpt_forskim", "passJetMult", "passPreSel_lowpt_forskim", "passPreSel"]
        else:
            all_cuts = ["passNoiseFilter", "passHLT", "passJetMult", "passPreSel"]

        if self.skim4b:
            all_cuts.append("passFourTag")

        for cut in all_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        if self.loosePtForSkim:
            selection = event.lumimask & event.passNoiseFilter & event.passJetMult_lowpt_forskim & event.passPreSel_lowpt_forskim
        else:
            selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.passPreSel

        if self.skim4b:
            selection = selection * event.fourTag

        if not config["isMC"]: selection = selection & event.passHLT


        return selection

    def preselect(self, event):
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        config = processor_config(processName, dataset, event)
        if config["isMC"] and self.mc_outlier_threshold is not None and "genWeight" in event.fields:
            return OutlierByMedian(self.mc_outlier_threshold)(event.genWeight)
