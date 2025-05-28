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
from base_class.hist import Collection, Fill
from jet_clustering.clustering_hist_templates import ClusterHists
from base_class.physics.object import Jet

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
        event["selFatJet"] = event.FatJet[event.FatJet.pt > 300]



##    # Create the PtEtaPhiMLorentzVectorArray
##    splittings_events = ak.zip(
##        {
##            "pt":   ak.Array([[v.pt   for v in sublist] for sublist in splittings]),
##            "eta":  ak.Array([[v.eta  for v in sublist] for sublist in splittings]),
##            "phi":  ak.Array([[v.phi  for v in sublist] for sublist in splittings]),
##            "mass": ak.Array([[v.mass for v in sublist] for sublist in splittings]),
##            "jet_flavor": ak.Array([[v.jet_flavor for v in sublist] for sublist in splittings]),
##            "btag_string": ak.Array([[v.btag_string for v in sublist] for sublist in splittings]),
##            "part_A": ak.zip(
##                {
##                    "pt":         ak.Array([[v.part_A.pt   for v in sublist] for sublist in splittings]),
##                    "eta":        ak.Array([[v.part_A.eta  for v in sublist] for sublist in splittings]),
##                    "phi":        ak.Array([[v.part_A.phi  for v in sublist] for sublist in splittings]),
##                    "mass":       ak.Array([[v.part_A.mass for v in sublist] for sublist in splittings]),
##                    "jet_flavor": ak.Array([[v.part_A.jet_flavor for v in sublist] for sublist in splittings]),
##                    "btag_string": ak.Array([[v.part_A.btag_string for v in sublist] for sublist in splittings]),
##                },
##                with_name="PtEtaPhiMLorentzVector",
##                behavior=vector.behavior
##            ),
##            "part_B": ak.zip(
##                {
##                    "pt":         ak.Array([[v.part_B.pt  for v in sublist] for sublist in splittings]),
##                    "eta":        ak.Array([[v.part_B.eta for v in sublist] for sublist in splittings]),
##                    "phi":        ak.Array([[v.part_B.phi for v in sublist] for sublist in splittings]),
##                    "mass":       ak.Array([[v.part_B.mass for v in sublist] for sublist in splittings]),
##                    "jet_flavor": ak.Array([[v.part_B.jet_flavor for v in sublist] for sublist in splittings]),
##                    "btag_string": ak.Array([[v.part_B.btag_string for v in sublist] for sublist in splittings]),
##                },
##                with_name="PtEtaPhiMLorentzVector",
##                behavior=vector.behavior
##            ),
##        },
##        with_name="PtEtaPhiMLorentzVector",
##        behavior=vector.behavior
##    )




        # Apply object selection (function does not remove events, adds content to objects)

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( "passNFatJets", ( ak.num(event.selFatJet) > 0 ) )
        ### add more selections, this can be useful

        list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "passNFatJets" ]


        #
        # Better Hists
        #

        # Hacking the tag variable
        event["fourTag"] = True
        event['tag'] = ak.zip({
            "fourTag": event.fourTag,
        })


        # Hack the region varable
        event["SR"] = True
        event["region"] = ak.zip({
            "SR": event.SR,
        })

        event["weight"] = 1.0

        event["passNFatJets"]  = (ak.num(event.selFatJet) > 0)

        fill = Fill(process=processName, year=year, weight="weight")
        histCuts = ["passNFatJets"]

        hist = Collection( process=[processName],
                           year=[year],
                           tag=["fourTag"],  # 3 / 4/ Other
                           region=['SR'],  # SR / SB / Other
                           **dict((s, ...) for s in histCuts)
                           )


        #
        # Jets
        #
        fill += Jet.plot(("fatJets", "Selected Fat Jets"),        "selFatJet",           skip=["deepjet_c"], bins={"pt": (50, 0, 1000)})

#        for _s_type in cleaned_splitting_name:
#            fill += ClusterHists( (f"splitting_{_s_type}", f"{_s_type} Splitting"), f"splitting_{_s_type}" )

        #
        # fill histograms
        #
        fill(event, hist)

        processOutput = {}

        output = hist.output | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
