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
from jet_clustering.clustering_hist_templates import ClusterHists, ClusterHistsDetailed
from jet_clustering.clustering   import cluster_bs, cluster_bs_fast
from jet_clustering.declustering import compute_decluster_variables, make_synthetic_event, get_list_of_splitting_types, clean_ISR, get_list_of_ISR_splittings, get_list_of_combined_jet_types, get_list_of_all_sub_splittings

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
            #   Make with ../.ci-workflows/synthetic-dataset-plot-job.sh
            # clustering_pdfs_file = "jet_clustering/jet-splitting-PDFs-0jet-00-01-00_5j/clustering_pdfs_vs_pT.yml",
            clustering_pdfs_file = "jet_clustering/jet-splitting-PDFs-00-03-00/clustering_pdfs_vs_pT.yml",
            #clustering_pdfs_file="jet_clustering/clustering_PDFs/clustering_pdfs_vs_pT.yml",
            do_declustering=False,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, "r"))
        self.clustering_pdfs = yaml.safe_load(open(clustering_pdfs_file, "r"))
        self.do_declustering = do_declustering

        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
            "passFourTag",
            #"pass0OthJets",
            #"pass1OthJets",
            #"pass2OthJets",
        ]

        self.histCuts = ["passPreSel"] #, "pass0OthJets", "pass1OthJets", "pass2OthJets"]


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

        #event['pass0OthJets'] = event.nJet_selected == 4
        #event['pass1OthJets'] = event.nJet_selected == 5
        #event['pass2OthJets'] = event.nJet_selected == 6
        #event['passMax1OthJets'] = event.nJet_selected < 6
        #event['passMax2OthJets'] = event.nJet_selected < 7
        #event['passMax4OthJets'] = event.nJet_selected < 9
        #selections.add("pass0OthJets",    event.pass0OthJets)
        #selections.add("pass1OthJets",    event.pass1OthJets)
        #selections.add("pass2OthJets",    event.pass2OthJets)
        #selections.add("passMax1OthJets", event.passMax1OthJets)
        #selections.add("passMax2OthJets", event.passMax2OthJets)
        #selections.add("passMax4OthJets", event.passMax4OthJets)
        allcuts.append("passFourTag")

        #allcuts.append("passMax1OthJets")
        #allcuts.append("passMax2OthJets")
        #allcuts.append("passMax4OthJets")
        #allcuts.append("pass2OthJets")

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
        notCanJet_sel = notCanJet[notCanJet.selected]

        notCanJet["isSelJet"] = 1 * ( (notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4) )  # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev["notCanJet_coffea"] = notCanJet
        selev["nNotCanJet"] = ak.num(selev.notCanJet_coffea)

        # print(selev.nJet_selected, selev.nNotCanJet, ak.num(notCanJet_sel),"\n")

        #
        # Build diJets, indexed by diJet[event,pairing,0/1]
        #
        #canJet = selev["canJet"]

        canJet["jet_flavor"] = "b"
        notCanJet_sel["jet_flavor"] = "j"
        selev["notCanJet_sel"] = notCanJet_sel

        jets_for_clustering = ak.concatenate([canJet, notCanJet_sel], axis=1)
        jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

        #
        #  To dump the testvectors
        #
        dumpTestVectors = False
        if dumpTestVectors:
            print(f'{chunk}\n\n')
            print(f'{chunk} self.input_jet_pt  = {[jets_for_clustering[iE].pt.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_eta  = {[jets_for_clustering[iE].eta.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_phi  = {[jets_for_clustering[iE].phi.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_mass  = {[jets_for_clustering[iE].mass.tolist() for iE in range(10)]}')
            print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering[iE].jet_flavor.tolist() for iE in range(10)]}')
            print(f'{chunk}\n\n')



        #clustered_jets, clustered_splittings = cluster_bs_fast(jets_for_clustering, debug=False)
        clustered_jets, clustered_splittings = cluster_bs(jets_for_clustering, debug=False)
        compute_decluster_variables(clustered_splittings)


        #
        #  get all splitting types that are used (ie: not pure ISR)
        #
        clustered_jets = clean_ISR(clustered_jets, clustered_splittings)

        cleaned_combined_types = get_list_of_combined_jet_types(clustered_jets)
        cleaned_split_types = []
        for _s in cleaned_combined_types:
            cleaned_split_types += get_list_of_all_sub_splittings(_s)


        #
        # Sort clusterings by type
        #
        for _s_type in cleaned_split_types:
            selev[f"splitting_{_s_type}"]   = clustered_splittings[clustered_splittings.jet_flavor == _s_type]

        # print(f'{chunk} all splitting types {all_split_types}\n')

        dumpTestVectors_bbj = False
        if dumpTestVectors_bbj:
            # bbj_mask = ak.num(selev["splitting_b(bj)"]) == 1
            # bbj_partA = selev["splitting_b(bj)"][bbj_mask].part_A
            # bbj_partB = selev["splitting_b(bj)"][bbj_mask].part_B
            #
            # if ak.sum(ak.num(selev["splitting_b(bj)"])) > 4:
            #     print(f'{chunk}\n\n')
            #     print(f'{chunk} self.input_jet_pt      = {[bbj_partA[iE].pt.tolist()         + bbj_partB[iE].pt.tolist()         for iE in range(5)]}')
            #     print(f'{chunk} self.input_jet_eta     = {[bbj_partA[iE].eta.tolist()        + bbj_partB[iE].eta.tolist()        for iE in range(5)]}')
            #     print(f'{chunk} self.input_jet_phi     = {[bbj_partA[iE].phi.tolist()        + bbj_partB[iE].phi.tolist()        for iE in range(5)]}')
            #     print(f'{chunk} self.input_jet_mass    = {[bbj_partA[iE].mass.tolist()       + bbj_partB[iE].mass.tolist()       for iE in range(5)]}')
            #     print(f'{chunk} self.input_jet_flavor  = {[bbj_partA[iE].jet_flavor.tolist() + bbj_partB[iE].jet_flavor.tolist() for iE in range(5)]}')
            #     print(f'{chunk}\n\n')

            print(f'{chunk} num splitting {ak.num(selev["splitting_b(bj)"])}')
            print(f'{chunk} mask {ak.num(selev["splitting_b(bj)"]) > 0}')
            bbj_mask = ak.num(selev["splitting_b(bj)"]) > 0
            jets_for_clustering_bbj = jets_for_clustering[bbj_mask]
            print(f'{chunk}\n\n')
            print(f'{chunk} self.input_jet_pt      = {[jets_for_clustering_bbj[iE].pt.tolist()         for iE in range(10)]}')
            print(f'{chunk} self.input_jet_eta     = {[jets_for_clustering_bbj[iE].eta.tolist()        for iE in range(10)]}')
            print(f'{chunk} self.input_jet_phi     = {[jets_for_clustering_bbj[iE].phi.tolist()        for iE in range(10)]}')
            print(f'{chunk} self.input_jet_mass    = {[jets_for_clustering_bbj[iE].mass.tolist()       for iE in range(10)]}')
            print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering_bbj[iE].jet_flavor.tolist() for iE in range(10)]}')
            print(f'{chunk}\n\n')



        #
        # writing out bb splitting for Chris Berman
        #
        # out_data = {}
        # out_data["pt_comb"]  = ak.flatten(selev["splitting_bb"].pt)
        # out_data["eta_comb"] = ak.flatten(selev["splitting_bb"].eta)
        # out_data["zA"] = ak.flatten(selev["splitting_bb"].zA)
        # out_data["thetaA"] = ak.flatten(selev["splitting_bb"].thetaA)
        # out_data["mA"] = ak.flatten(selev["splitting_bb"].mA)
        # out_data["mB"] = ak.flatten(selev["splitting_bb"].mB)
        # out_data["decay_phi"] = ak.flatten(selev["splitting_bb"].decay_phi)
        #
        # for out_k, out_v in out_data.items():
        #     processOutput[out_k] = {}
        #     processOutput[out_k][event.metadata['dataset']] = list(out_v)


        #
        #  Declustering
        #
        if self.do_declustering:

            # clustered_jets = clean_ISR(clustered_jets, clustered_splittings)

            #
            # Declustering
            #

            #
            #  Read in the pdfs
            #

            declustered_jets = make_synthetic_event(clustered_jets, self.clustering_pdfs)

            is_b_mask = declustered_jets.jet_flavor == "b"
            canJet_re = declustered_jets[is_b_mask]

            canJet_re["puId"] = 7
            canJet_re["jetId"] = 7 # selev.Jet.puId[canJet_idx]
            canJet_re["btagDeepFlavB"] = 1.0 # Set bs to 1 and ls to 0


            notCanJet_sel_re = declustered_jets[~is_b_mask]
            notCanJet_sel_re["puId"] = 7
            notCanJet_sel_re["jetId"] = 7 # selev.Jet.puId[canJet_idx]
            notCanJet_sel_re["btagDeepFlavB"] = 0 # Set bs to 1 and ls to 0


            selev["canJet_re"] = canJet_re
            selev["notCanJet_sel_re"] = notCanJet_sel_re

            #
            #  Recluster
            #
            jets_for_clustering = ak.concatenate([canJet_re, notCanJet_sel_re], axis=1)
            jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

            clustered_jets_reclustered, clustered_splittings_reclustered = cluster_bs(jets_for_clustering, debug=False)
            compute_decluster_variables(clustered_splittings_reclustered)

            # all_split_types_re = get_list_of_splitting_types(clustered_splittings_reclustered)
            # # ISR_splittings_re  = get_list_of_ISR_splittings(all_split_types_re)
            # ISR_splittings_re = [] # Hack Save all splitting for now
            # all_split_types_re = [item for item in all_split_types_re if item not in ISR_splittings_re]

            for _s_type in cleaned_split_types:
                selev[f"splitting_{_s_type}_re"]  = clustered_splittings_reclustered[clustered_splittings_reclustered.jet_flavor == _s_type]

            # print(f'{chunk} all splitting_re types {all_split_types_re}\n')

            debug_bbj = False
            if debug_bbj:
                bbj_mask = ak.num(selev["splitting_b(bj)_re"]) > 0
                #bbj_partA = selev["splitting_b(bj)_re"][bbj_mask].part_A

                selev_bbjj = selev[bbj_mask]

                bbj_partB_large_mass = selev_bbjj["splitting_b(bj)_re"].part_B.mass > 50
                print(f'{chunk} mass {selev_bbjj["splitting_b(bj)_re"].part_B.mass}')
                print(f'{chunk} have large {bbj_partB_large_mass}')
                print(f'{chunk} any {ak.any(bbj_partB_large_mass, axis=1)}')

                large_bbj_mb_event_mask = ak.any(bbj_partB_large_mass, axis=1)

                selev_large_bbj = selev_bbjj[large_bbj_mb_event_mask]

                print(f'{chunk} partB mass {selev_large_bbj["splitting_b(bj)_re"].part_B.mass}\n')
                print(f'{chunk} partB flav {selev_large_bbj["splitting_b(bj)_re"].part_B.jet_flavor}\n')
                print(f'{chunk} partB pt {selev_large_bbj["splitting_b(bj)_re"].part_B.pt}\n')
                print(f'{chunk} partB eta {selev_large_bbj["splitting_b(bj)_re"].part_B.eta}\n')


                print(f'{chunk} partA mass {selev_large_bbj["splitting_b(bj)_re"].part_A.mass}\n')
                print(f'{chunk} partA falv {selev_large_bbj["splitting_b(bj)_re"].part_A.jet_flavor}\n')
                print(f'{chunk} partA pt {selev_large_bbj["splitting_b(bj)_re"].part_A.pt}\n')
                print(f'{chunk} partA eta {selev_large_bbj["splitting_b(bj)_re"].part_A.eta}\n')

            dumpTestVectors = False
            if dumpTestVectors:
                print(f'{chunk}\n\n')
                print(f'{chunk} self.input_jet_pt  = {[jets_for_clustering[iE].pt.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_eta  = {[jets_for_clustering[iE].eta.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_phi  = {[jets_for_clustering[iE].phi.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_mass  = {[jets_for_clustering[iE].mass.tolist() for iE in range(10)]}')
                print(f'{chunk} self.input_jet_flavor  = {[jets_for_clustering[iE].jet_flavor.tolist() for iE in range(10)]}')
                print(f'{chunk}\n\n')




        selev["region"] = 0b10

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[selections.all(*allcuts)]

        self._cutFlow.fill("passFourTag", selev )
        #self._cutFlow.fill("pass0OthJets",selev )
        #self._cutFlow.fill("pass1OthJets",selev )
        #self._cutFlow.fill("pass2OthJets",selev )

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
        # fill += hist.add( "nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")) )
        # fill += hist.add( "nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")), )

        #
        # Jets
        #
        fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])
        # fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=["deepjet_c"])
        # fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=["deepjet_c"])
        # fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=["deepjet_c"])

        # fill += Jet.plot(("notCanJet_sel", "Higgs Candidate Jets"), "notCanJet_sel",           skip=["deepjet_c"])
        # if self.do_declustering:
        #     fill += Jet.plot(("canJets_re", "Higgs Candidate Jets"), "canJet_re",           skip=["deepjet_c"])
        #     fill += Jet.plot(("notCanJet_sel_re", "Higgs Candidate Jets"), "notCanJet_sel_re",           skip=["deepjet_c"])

        #
        #  Make Jet Hists
        #
        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]

        for iJ in range(4):
            fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )


        for _s_type in cleaned_split_types:
            fill += ClusterHists( (f"splitting_{_s_type}", f"{_s_type} Splitting"), f"splitting_{_s_type}" )

        fill += ClusterHistsDetailed( (f"detailed_splitting_bb",    f"bb Splitting"),    f"splitting_bb"    )
        fill += ClusterHistsDetailed( (f"detailed_splitting_bj",    f"bj Splitting"),    f"splitting_bj"    )
        #fill += ClusterHistsDetailed( (f"detailed_splitting_jj",    f"jj Splitting"),    f"splitting_jj"    )
        fill += ClusterHistsDetailed( (f"detailed_splitting_(bj)b", f"(bj)b Splitting"), f"splitting_(bj)b" )

        if self.do_declustering:
            for _s_type in cleaned_split_types:
                fill += ClusterHists( (f"splitting_{_s_type}_re", f"${_s_type} Splitting"), f"splitting_{_s_type}_re" )


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
