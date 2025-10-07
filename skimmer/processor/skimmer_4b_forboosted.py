import yaml
import logging
import awkward as ak
import numpy as np
from coffea.util import load
from src.skimmer.picoaod import PicoAOD #, fetch_metadata, resize
from coffea4bees.analysis.helpers.cutflow import cutflow_4b

class Skimmer(PicoAOD):
    def __init__(
            self, 
            file_wEvents="", 
            corrections_metadata: dict = None, 
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.file_wEvents = load(file_wEvents)
        self.corrections_metadata = corrections_metadata
        self._cutFlow = cutflow_4b()

    def select(self, events):

        isMC    = True if events.run[0] == 1 else False
        year    = events.metadata['year']
        dataset = events.metadata['dataset']

        resolved_events = self.file_wEvents['event'][f"{dataset}"]
        resolved_selection_SR = np.isin( events.event.to_numpy(), resolved_events )

        return resolved_selection_SR
