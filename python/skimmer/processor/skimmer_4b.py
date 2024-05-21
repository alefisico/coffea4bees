import yaml
from skimmer.processor.picoaod import PicoAOD, fetch_metadata, resize
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b
from coffea.analysis_tools import Weights, PackedSelection
import numpy as np

class Skimmer(PicoAOD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrections_metadata = yaml.safe_load(open('analysis/metadata/corrections.yml', 'r'))
        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passPreSel",
        ]



    def select(self, event):

        isMC    = True if event.run[0] == 1 else False
        year    = event.metadata['year']
        dataset = event.metadata['dataset']

        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )
        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year]  )

        weights = Weights(len(event), storeIndividual=True)

        #
        # general event weights
        #
        if isMC:
            weights.add( "genweight_", event.genWeight )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( 'passJetMult', event.passJetMult )
        selections.add( "passPreSel", event.passPreSel)

        event["weight"] = weights.weight()

        self._cutFlow.fill( "all",             event[selections.all(*["lumimask"])],                   allTag=True )
        self._cutFlow.fill( "passNoiseFilter", event[selections.all(*["lumimask","passNoiseFilter"])], allTag=True )
        self._cutFlow.fill( "passHLT",         event[selections.all(*["lumimask","passNoiseFilter", "passHLT"])], allTag=True )
        self._cutFlow.fill( "passJetMult",     event[selections.all(*["lumimask","passNoiseFilter", "passHLT", "passJetMult"])], allTag=True )
        self._cutFlow.fill( "passPreSel",      event[selections.all(*["lumimask","passNoiseFilter", "passHLT", "passJetMult", "passPreSel"])], allTag=True )

        selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.passPreSel
        if not isMC: selection = selection & event.passHLT

        return selection
