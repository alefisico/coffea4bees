import yaml
from skimmer.processor.picoaod import PicoAOD, fetch_metadata, resize
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b
from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from analysis.helpers.common import init_jet_factory
from copy import copy
import logging
import awkward as ak

class Skimmer(PicoAOD):
    def __init__(self, loosePtForSkim=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loosePtForSkim = loosePtForSkim
        self.corrections_metadata = yaml.safe_load(open('analysis/metadata/corrections.yml', 'r'))
        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult_lowpt_forskim",
            "passJetMult",
            "passPreSel_lowpt_forskim",
            "passPreSel",
        ]



    def select(self, event):

        isMC    = True if event.run[0] == 1 else False
        year    = event.metadata['year']
        dataset = event.metadata['dataset']


        #
        #  Nominal config (...what we would do for data)
        #
        cut_on_lumimask         = True
        cut_on_HLT_decision     = True
        do_lepton_jet_cleaning  = True

        if isMC:
            cut_on_lumimask     = False
            cut_on_HLT_decision = False


        event = apply_event_selection_4b( event, self.corrections_metadata[year], cut_on_lumimask=cut_on_lumimask )


        juncWS = [ self.corrections_metadata[year]["JERC"][0].replace("STEP", istep)
                   for istep in ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"] ] #+ self.corrections_metadata[year]["JERC"][2:]

        #old_jets = copy(event.Jet)
        jets = init_jet_factory(juncWS, event, isMC)
        event["Jet"] = jets


        event = apply_object_selection_4b( event, self.corrections_metadata[year], doLeptonRemoval=do_lepton_jet_cleaning, loosePtForSkim=self.loosePtForSkim  )

        weights = Weights(len(event), storeIndividual=True)

        #
        # general event weights
        #
        if isMC:
            weights.add( "genweight_", event.genWeight )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if cut_on_HLT_decision else np.full(len(event), True)  ) )
        if self.loosePtForSkim:
            selections.add( 'passJetMult_lowpt_forskim', event.passJetMult_lowpt_forskim )
        selections.add( 'passJetMult',   event.passJetMult )
        if self.loosePtForSkim:
            selections.add( "passPreSel_lowpt_forskim",  event.passPreSel_lowpt_forskim)
        selections.add( "passPreSel",    event.passPreSel)

        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        if self.loosePtForSkim:
            all_cuts = ["passNoiseFilter", "passHLT", "passJetMult_lowpt_forskim", "passJetMult", "passPreSel_lowpt_forskim", "passPreSel"]
        else:
            all_cuts = ["passNoiseFilter", "passHLT", "passJetMult", "passPreSel"]

        for cut in all_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        if self.loosePtForSkim:
            selection = event.lumimask & event.passNoiseFilter & event.passJetMult_lowpt_forskim & event.passPreSel_lowpt_forskim
        else:
            selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.passPreSel
        if not isMC: selection = selection & event.passHLT

        return selection
