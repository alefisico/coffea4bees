import time
import gc
import awkward as ak
import numpy as np
import correctionlib
import yaml
import warnings
import uproot

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.util import load
from coffea.analysis_tools import Weights, PackedSelection

from base_class.hist import Collection, Fill
from base_class.physics.object import LorentzVector, Jet, Muon, Elec
#from analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists
from analysis.helpers.clustering_hist_templates import ClusterHists
from analysis.helpers.topCandReconstruction import dumpTopCandidateTestVectors
from analysis.helpers.clustering import cluster_bs, compute_decluster_variables, cluster_bs_fast

from analysis.helpers.cutflow import cutFlow
from analysis.helpers.FriendTreeSchema import FriendTreeSchema

from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.common import init_jet_factory, apply_btag_sf, update_events

from analysis.helpers.selection_basic_4b import (
    apply_event_selection_4b,
    apply_object_selection_4b
)

import logging

from base_class.root import TreeReader, Chunk

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        threeTag=False,
        corrections_metadata="analysis/metadata/corrections.yml",
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, "r"))

        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
            "passFourTag",
            "pass0OthJets",
        ]

        self.histCuts = ["passPreSel"]


    def process(self, event):

        tstart = time.time()
        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        nEvent = len(event)

        logging.debug(fname)
        logging.debug(f'{chunk}Process {nEvent} Events')

        #
        # Event selection
        #
        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )

        juncWS = [ self.corrections_metadata[year]["JERC"][0].replace("STEP", istep)
                   for istep in ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"] ] #+ self.corrections_metadata[year]["JERC"][2:]

        jets = init_jet_factory(juncWS, event, isMC)

        shifts = [({"Jet": jets}, None)]

        weights = Weights(len(event), storeIndividual=True)
        list_weight_names = []

        logging.debug(f"weights event {weights.weight()[:10]}")
        logging.debug(f"Weight Statistics {weights.weightStatistics}")

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year] )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        selections.add( 'passJetMult', event.passJetMult )
        allcuts = [ 'lumimask', 'passNoiseFilter', 'passHLT', 'passJetMult' ]
        event['weight'] = weights.weight()   ### this is for _cutflow

        #
        #  Cut Flows
        #
        processOutput = {}

        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = {
            'nEvent' : nEvent,
            'genWeights': np.sum(event.genWeight) if isMC else nEvent

        }

        self._cutFlow = cutFlow(self.cutFlowCuts)
        self._cutFlow.fill( "all", event[selections.require(lumimask=True)], allTag=True)
        self._cutFlow.fill( "passNoiseFilter", event[selections.require(lumimask=True, passNoiseFilter=True)], allTag=True)
        self._cutFlow.fill( "passHLT", event[ selections.require( lumimask=True, passNoiseFilter=True, passHLT=True ) ], allTag=True, )
        self._cutFlow.fill( "passJetMult", event[ selections.all(*allcuts)], allTag=True )


        #
        # Preselection: keep only three or four tag events
        #
        #selections.add("passPreSel", event.passPreSel)
        selections.add("passFourTag", event.fourTag)

        event['pass0OthJets'] = event.nJet_selected == 4
        selections.add("pass0OthJets", event.pass0OthJets)
        allcuts.append("passFourTag")
        allcuts.append("pass0OthJets")

        selev = event[selections.all(*allcuts)]

        # logging.info( f"\n {chunk} Event:  nSelJets {selev['nJet_selected']}\n")

        #
        # Build and select boson candidate jets with bRegCorr applied
        #
        sorted_idx = ak.argsort( selev.Jet.btagDeepFlavB * selev.Jet.selected, axis=1, ascending=False )
        canJet_idx = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]
        canJet = selev.Jet[canJet_idx]

        # apply bJES to canJets
        canJet = canJet * canJet.bRegCorr
        canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
        canJet["btagDeepFlavB"] = selev.Jet.btagDeepFlavB[canJet_idx]
        canJet["puId"] = selev.Jet.puId[canJet_idx]
        canJet["jetId"] = selev.Jet.puId[canJet_idx]
        if isMC:
            canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]
        canJet["calibration"] = selev.Jet.calibration[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        selev["canJet"] = canJet

        #
        #  Should be a better way to do this...
        #
        selev["canJet0"] = canJet[:, 0]
        selev["canJet1"] = canJet[:, 1]
        selev["canJet2"] = canJet[:, 2]
        selev["canJet3"] = canJet[:, 3]

        # selev['v4j', 'n'] = 1
        # print(selev.v4j.n)
        # selev['Jet', 'canJet'] = False
        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        notCanJet["isSelJet"] = 1 * ( (notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4) )  # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev["notCanJet_coffea"] = notCanJet
        selev["nNotCanJet"] = ak.num(selev.notCanJet_coffea)

        #
        # Build diJets, indexed by diJet[event,pairing,0/1]
        #
        #canJet = selev["canJet"]

        canJet["jet_flavor"] = "b"
        #clustered_jets, clustered_splittings = cluster_bs_fast(canJet, debug=False)
        clustered_jets, clustered_splittings = cluster_bs(canJet, debug=False)
        compute_decluster_variables(clustered_splittings)

        selev["splitting_g_bb"]   = clustered_splittings[clustered_splittings.jet_flavor == "g_bb"]
        selev["splitting_bstar"] = clustered_splittings[clustered_splittings.jet_flavor == "bstar"]

        # dumpTopCandidateTestVectors(selev, logging, chunk, 10)
        selev["region"] = 0b10


        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[selections.all(*allcuts)]

        self._cutFlow.fill("passFourTag", selev )
        self._cutFlow.fill("pass0OthJets",selev )

        self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        #
        # Hists
        #

        fill = Fill(process=processName, year=year, weight="weight")

        hist = Collection( process=[processName],
                           year=[year],
                           tag=[3, 4, 0],  # 3 / 4/ Other
                           region=[2, 1, 0],  # SR / SB / Other
                           **dict((s, ...) for s in self.histCuts)
                           )

        #
        # To Add
        #
        fill += hist.add( "nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")) )
        fill += hist.add( "nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")), )

        #
        # Jets
        #
        fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])
        fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=["deepjet_c"])
        fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=["deepjet_c"])
        fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=["deepjet_c"])

        #
        #  Make Jet Hists
        #
        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]

        for iJ in range(4):
            fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )

        fill += ClusterHists( ("gbbs", "$g_{bb}$ Splitting"), "splitting_g_bb" )
        fill += ClusterHists( ("bstars", "$b^*$ Splitting"), "splitting_bstar" )

        #
        # fill histograms
        #
        # fill.cache(selev)
        fill(selev, hist)

        garbage = gc.collect()
        # print('Garbage:',garbage)


        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f"{chunk}{nEvent/elapsed:,.0f} events/s")

        output = hist.output | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
