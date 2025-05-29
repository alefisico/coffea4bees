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

        logging.info(fname)
        logging.info(f'Process {nEvent} Events')

        #
        # Event selection
        #
        event = apply_event_selection( event,
                                        self.corrections_metadata[year],
                                        cut_on_lumimask=True
                                        )



        selFatJet = event.FatJet[event.FatJet.pt > 300]
        selFatJet = selFatJet[ak.num(selFatJet.subjets, axis=2) > 1]

        print(f" fields FatJets: {selFatJet.fields}")
        print(f" fields nSubJets: {selFatJet.subjets.fields}")
        print(f" nSubJets: {ak.num(selFatJet.subjets, axis=1)}")

        #selFatJet = selFatJet[ak.num(selFatJet.subjets) > 1]
        event["selFatJet"] = selFatJet


        #  Cehck How often do we have >=2 Fat Jets?
        event["passNFatJets"]  = (ak.num(event.selFatJet) == 2)








        # Apply object selection (function does not remove events, adds content to objects)

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( "passNFatJets",  event.passNFatJets )
        ### add more selections, this can be useful


        #list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "passNFatJets" ]
        list_of_cuts = [ "passNFatJets" ]
        analysis_selections = selections.all(*list_of_cuts)
        selev = event[analysis_selections]

        #
        # Event selection
        #
        #event["selFatJet"] = event.FatJet[event.FatJet. > 1]

        print(f"Number of selected Fat Jets: {ak.num(selev.selFatJet)}")
        print(f" Any passNFatJets: {ak.any(selev.passNFatJets)}")
        print(f" Any passHLT: {ak.any(selev.passHLT)}")
        print(f" FatJet pt: {selev.selFatJet.pt}")

        print(f" nSubJets: {ak.num(selev.selFatJet.subjets, axis=2)}")
        print(f" subjet pt: {selev.selFatJet.pt[0:10]}")

        print(f" FatJet subjet pt: {selev.selFatJet.subjets.pt[0]}")
        print(f" FatJet subjet0 pt: {selev.selFatJet[:,0].subjets.pt}")
        print(f" FatJet subjet0_0 pt: {selev.selFatJet[:,0].subjets[:,0].pt}")
        print(f" FatJet subjet0_1 pt: {selev.selFatJet[:,0].subjets[:,1].pt}")
        print(f" FatJet subjet1 pt: {selev.selFatJet[:,1].subjets.pt}")
        #print("SubJet 0",ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet.subjets[:,0]]))
        #print("SubJet 1",ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet.subjets[:,1]]))

        print( "v" , ak.Array([[v   for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,0]]) )

        print( ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,0]]) )

#        # Create the PtEtaPhiMLorentzVectorArray
#        fat_jet_splittings_events = ak.zip(
#            {
#                "pt":   ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet[:,0]]),
#                "eta":  ak.Array([[v.eta  for v in sublist] for sublist in selev.selFatJet[:,0]]),
#                "phi":  ak.Array([[v.phi  for v in sublist] for sublist in selev.selFatJet[:,0]]),
#                "mass": ak.Array([[v.mass for v in sublist] for sublist in selev.selFatJet[:,0]]),
#                #"jet_flavor": ak.Array([[v.jet_flavor for v in sublist] for sublist in splittings]),   # "bb"
#                #"btag_string": ak.Array([[v.btag_string for v in sublist] for sublist in splittings]),  # str(particleNet_HbbvsQCD)
#                "part_A": ak.zip(
#                    {
#                        "pt":         ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,0]]),
#                        "eta":        ak.Array([[v.eta  for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,0]]),
#                        "phi":        ak.Array([[v.phi  for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,0]]),
#                        "mass":       ak.Array([[v.mass for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,0]]),
#                        #"jet_flavor": ak.Array([[v.part_A.jet_flavor for v in sublist] for sublist in splittings]), # "b"
#                        #"btag_string": ak.Array([[v.part_A.btag_string for v in sublist] for sublist in splittings]),  # str(btagDeepB)
#                    },
#                    with_name="PtEtaPhiMLorentzVector",
#                    behavior=vector.behavior
#                ),
#                "part_B": ak.zip(
#                    {
#                        "pt":         ak.Array([[v.pt   for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,1]]),
#                        "eta":        ak.Array([[v.eta  for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,1]]),
#                        "phi":        ak.Array([[v.phi  for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,1]]),
#                        "mass":       ak.Array([[v.mass for v in sublist] for sublist in selev.selFatJet[:,0].subjets[:,1]]),
#                        #"jet_flavor": ak.Array([[v.part_B.jet_flavor for v in sublist] for sublist in splittings]),   # "b"
#                        #"btag_string": ak.Array([[v.part_B.btag_string for v in sublist] for sublist in splittings]), # str(btagDeepB)
#                    },
#                    with_name="PtEtaPhiMLorentzVector",
#                    behavior=vector.behavior
#                ),
#            },
#            with_name="PtEtaPhiMLorentzVector",
#            behavior=vector.behavior
#        )



        #
        # Better Hists
        #

        # Hacking the tag variable
        selev["fourTag"] = True
        selev['tag'] = ak.zip({
            "fourTag": selev.fourTag,
        })


        # Hack the region varable
        selev["SR"] = True
        selev["region"] = ak.zip({
            "SR": selev.SR,
        })

        selev["weight"] = 1.0



        fill = Fill(process=processName, year=year, weight="weight")
        histCuts = ["passNFatJets"]

        hist = Collection( process=[processName],
                           year=[year],
                           tag=["fourTag"],  # 3 / 4/ Other
                           region=['SR'],  # SR / SB / Other
                           **dict((s, ...) for s in histCuts)
                           )


        print(f" SubJets: {selev.selFatJet.subjets}")
        print(f" SubJets fields: {selev.selFatJet.subjets.fields}")
        selev["selFatJet_subjets"] = selev.selFatJet.subjets
        #print(f" SubJets pt: {selev.selFatJet_subjets.pt}")

        #
        # Jets
        #
        fill += Jet.plot(("fatJets", "Selected Fat Jets"),        "selFatJet",           skip=["deepjet_c"], bins={"pt": (50, 0, 1000)})

        # print(f" SubJets pt {selev.selFatJet_subjets.pt[0:5]}\n")
        # fill += Jet.plot(("subJets", "Selected Fat Jet SubJet"),   "selFatJet_subjets",  skip=["deepjet_c","deepjet_b","id_pileup","id_jet","n"], bins={"pt": (50, 0, 1000)})

#        for _s_type in cleaned_splitting_name:
#            fill += ClusterHists( (f"splitting_{_s_type}", f"{_s_type} Splitting"), f"splitting_{_s_type}" )

        #
        # fill histograms
        #
        fill(selev, hist)

        processOutput = {}

        output = hist.output | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
