import time
import awkward as ak
import numpy as np
import yaml
import warnings

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection
import hist

from analysis.helpers.event_selection import apply_event_selection

import logging

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            corrections_metadata="analysis/metadata/corrections.yml",
            **kwargs
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, "r"))

    def process(self, event):

        ### Some useful variables
        tstart = time.time()
        fname   = event.metadata['filename']
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        nEvent = len(event)

        logging.debug(fname)
        logging.debug(f'Process {nEvent} Events')

        #
        # Event selection
        #
        event = apply_event_selection( event,
                                        self.corrections_metadata[year],
                                        cut_on_lumimask=True
                                        )

        #
        # Event selection
        #
        jets = event.FatJet[event.FatJet.pt > 200]

        # Apply object selection (function does not remove events, adds content to objects)
        
        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( "num_jets", ( ak.num(jets) > 0 ) )
        ### add more selections, this can be useful

        list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "num_jets" ]
        
        #
        # Histograms
        #
        process_axis = hist.axis.StrCategory([], name="process", label="Process", growth=True)
        sel_axis = hist.axis.StrCategory([], name="selection", label="Selection", growth=True)
        year_axis = hist.axis.StrCategory([], name="year", label="Year", growth=True)
        jet_pt = hist.axis.Regular(100, 0., 1000., name="jet_pt", label="Jet Pt [GeV]")

        output = {
            'hists' : {
                "jet_pt": hist.Hist(process_axis, sel_axis, year_axis, jet_pt),
            }
        }

        output["hists"]["jet_pt"].fill(
            process=processName,
            selection="all",
            year=year,
            jet_pt=ak.flatten(jets.pt[selections.all(*list_of_cuts)]),
        )
        
        return output

    def postprocess(self, accumulator):
        return accumulator
