import yaml
from skimmer.processor.picoaod import PicoAOD, fetch_metadata, resize
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b


class Skimmer(PicoAOD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrections_metadata = yaml.safe_load(open('analysis/metadata/corrections.yml', 'r'))

    def select(self, events):

        isMC    = True if events.run[0] == 1 else False
        year    = events.metadata['year']
        dataset = events.metadata['dataset']

        events = apply_event_selection_4b( events, isMC, self.corrections_metadata[year] )
        events = apply_object_selection_4b( events, year, isMC, dataset, self.corrections_metadata[year]  )

        selection = events.lumimask & events.passNoiseFilter & events.passJetMult & events.passPreSel
        if not isMC: selection = selection & events.passHLT

        return selection

