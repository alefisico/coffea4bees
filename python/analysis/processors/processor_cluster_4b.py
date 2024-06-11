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
from analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists
from analysis.helpers.topCandReconstruction import dumpTopCandidateTestVectors

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

        event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)
        selev = event[selections.all(*allcuts)]

        logging.info( f"\n {chunk} Event:  nSelJets {selev['nJet_selected']}\n")
        
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
        canJet = selev["canJet"]

        dRs = []
        jet_pairs_idx_i = []
        jet_pairs_idx_j = []
        jet_pairs_idx = []
        
        for i in range(4):
            for j in range(i+1,4):
                print(i,j)
                #_diJet = canJet[:, i] + canJet[:, j]
                _dR = canJet[:, i].delta_r(canJet[:, j])
                dRs.append(_dR)
                jet_pairs_idx_i.append( i )
                jet_pairs_idx_j.append( j )
                jet_pairs_idx  .append( (i, j) ) 

        import fastjet
        jetdef = fastjet.JetDefinition(fastjet.cambridge_algorithm, 4*np.pi)
        cluster = fastjet.ClusterSequence(canJet, jetdef)
        print(cluster)
        print(dir(cluster))
        print()
        print(f'canJet 0 "pt" : {canJet[0, 0].pt}, "eta" : {canJet[0, 0].eta}, "phi" : {canJet[0, 0].phi}, "E" : {canJet[0, 0].E} ')
        print(f'canJet 1 "pt" : {canJet[0, 1].pt}, "eta" : {canJet[0, 1].eta}, "phi" : {canJet[0, 1].phi}, "E" : {canJet[0, 1].E} ')
        print(f'canJet 2 "pt" : {canJet[0, 2].pt}, "eta" : {canJet[0, 2].eta}, "phi" : {canJet[0, 2].phi}, "E" : {canJet[0, 2].E} ')
        print(f'canJet 3 "pt" : {canJet[0, 3].pt}, "eta" : {canJet[0, 3].eta}, "phi" : {canJet[0, 3].phi}, "E" : {canJet[0, 3].E} ')        

        print(f'{selev.Jet[0]}')
        print(f'{selev.Jet[0][0]}')
        print(f'{type(selev.Jet[0][0])}')
        print()

        print(f'{len(selev)}')
        print()
        dumpTopCandidateTestVectors(selev, logging, chunk, 10)
        
        print(f' n_exclusive_jets {cluster.n_exclusive_jets()[0]}')
        print(cluster.inclusive_jets()[0])
        print(cluster.constituents()[0])
        print()
#        pairing = [jet_pairs_idx_i, jet_pairs_idx_j]
#        diJets = canJet[:, pairing[0]] + canJet[:, pairing[1]]
#        
#        logging.info( f"\n {chunk} can dRs {dRs[0]} {jet_pairs_idx[0]} \n")
#        logging.info( f"\n {chunk} can dRs {dRs[1]} {jet_pairs_idx[1]} \n")
#        logging.info( f"\n {chunk} can dRs {dRs[2]} {jet_pairs_idx[2]} \n")
#        logging.info( f"\n {chunk} can dRs {dRs[3]} {jet_pairs_idx[3]} \n")
#        logging.info( f"\n {chunk} can dRs {dRs[4]} {jet_pairs_idx[4]} \n")
#        logging.info( f"\n {chunk} can dRs {dRs[5]} {jet_pairs_idx[5]} \n")
#
#
#        
#        all_dRs = canJet[:, pairing[0]].delta_r(canJet[:, pairing[1]])
#        logging.info( f"\n {chunk} all can dRs {all_dRs} \n")
#        logging.info( f"\n {chunk} shape {all_dRs.ndim} \n")
#        dR_sorted_idx = ak.argsort( all_dRs, axis=1, ascending=True )
#        logging.info( f"\n {chunk} sorted idx {dR_sorted_idx} \n")
#

        
        #logging.info( f"\n {chunk} can dRs {len(dRs[3])} \n")
        
                
        pairing = [([0, 2], [0, 1], [0, 1]), ([1, 3], [2, 3], [3, 2])]
        diJet = canJet[:, pairing[0]] + canJet[:, pairing[1]]

        #logging.info( f"\n {chunk} {canJet[:, pairing[0]][0][0]} + {canJet[:, pairing[1]][0][0]}")
        #logging.info( f"\n {chunk} {len(canJet[:, pairing[0]][0])}")
        #logging.info( f"\n {chunk} {len(diJet[0])} ")
        #logging.info( f"\n {chunk} di-jet0 pT {diJet[0][0].pt} ")


        diJet["st"] = canJet[:, pairing[0]].pt + canJet[:, pairing[1]].pt
        diJet["dr"] = canJet[:, pairing[0]].delta_r(canJet[:, pairing[1]])
        diJet["dphi"] = canJet[:, pairing[0]].delta_phi(canJet[:, pairing[1]])
        diJet["lead"] = canJet[:, pairing[0]]
        diJet["subl"] = canJet[:, pairing[1]]
        # Sort diJets within views to be lead st, subl st
        diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
        diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]
        # Now indexed by diJet[event,pairing,lead/subl st]

        # Compute diJetMass cut with independent min/max for lead/subl
        minDiJetMass = np.array([[[52, 50]]])
        maxDiJetMass = np.array([[[180, 173]]])
        diJet["passDiJetMass"] = (minDiJetMass < diJet.mass) & ( diJet.mass < maxDiJetMass )

        # Compute MDRs
        min_m4j_scale = np.array([[360, 235]])
        min_dr_offset = np.array([[-0.5, 0.0]])
        max_m4j_scale = np.array([[650, 650]])
        max_dr_offset = np.array([[0.5, 0.7]])
        max_dr = np.array([[1.5, 1.5]])

        #
        # Compute consistency of diJet masses with boson masses
        #
        mZ = 91.0
        mH = 125.0
        st_bias = np.array([[[1.02, 0.98]]])
        cZ = mZ * st_bias
        cH = mH * st_bias

        diJet["xZ"] = (diJet.mass - cZ) / (0.1 * diJet.mass)
        diJet["xH"] = (diJet.mass - cH) / (0.1 * diJet.mass)

        #
        # Build quadJets
        #
        seeds = np.array(event.event)[[0, -1]].view(np.ulonglong)
        randomstate = np.random.Generator(np.random.PCG64(seeds))
        quadJet = ak.zip( { "lead": diJet[:, :, 0],
                            "subl": diJet[:, :, 1],
                            "close": diJetDr[:, :, 0],
                            "other": diJetDr[:, :, 1],
                            "passDiJetMass": ak.all(diJet.passDiJetMass, axis=2),
                            "random": randomstate.uniform(
                                low=0.1, high=0.9, size=(diJet.__len__(), 3)
                            ), } )

        quadJet["dr"] = quadJet["lead"].delta_r(quadJet["subl"])
        quadJet["dphi"] = quadJet["lead"].delta_phi(quadJet["subl"])
        quadJet["deta"] = quadJet["lead"].eta - quadJet["subl"].eta




        #
        # Compute Signal Regions
        #
        quadJet["xZZ"] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
        quadJet["xHH"] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
        quadJet["xZH"] = np.sqrt( np.minimum( quadJet.lead.xH**2 + quadJet.subl.xZ**2, quadJet.lead.xZ**2 + quadJet.subl.xH**2, ) )

        max_xZZ = 2.6
        max_xZH = 1.9
        max_xHH = 1.9


        #
        #  Build the close dR and other quadjets
        #    (There is Probably a better way to do this ...
        #
        arg_min_close_dr = np.argmin(quadJet.close.dr, axis=1)
        arg_min_close_dr = arg_min_close_dr.to_numpy()
        selev["quadJet_min_dr"] = quadJet[ np.array(range(len(quadJet))), arg_min_close_dr ]

        #
        # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
        #

        selev["diJet"] = diJet
        selev["quadJet"] = quadJet
        selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)


        #
        # Example of how to write out event numbers
        #
        #  passSR = (selev["quadJet_selected"].SR)
        #  passSR = (selev["SR"])
        #
        # out_data = {}
        # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
        # out_data["event"  ] = selev["event"][passSR]
        # out_data["run"    ] = selev["run"][passSR]
        #
        # debug_mask = ~event.passJetMult
        # debug_mask = ((event["event"] == 66688  ) |
        #               (event["event"] == 249987 ) |
        #               (event["event"] == 121603 ) |
        #               (event["event"] == 7816   ) |
        #               (event["event"] == 25353  ) |
        #               (event["event"] == 165389 ) |
        #               (event["event"] == 293138 ) |
        #               (event["event"] == 150164 ) |
        #               (event["event"] == 262806 ) |
        #               (event["event"] == 281111 ) )
        #
        # out_data["debug_event"  ] = event["event"][debug_mask]
        # out_data["debug_run"    ] = event["run"][debug_mask]
        # out_data["debug_jet_pt"    ] = event.Jet[event.Jet.selected_eta].pt[debug_mask].to_list()
        # out_data["debug_jet_eta"   ] = event.Jet[event.Jet.selected_eta].eta[debug_mask].to_list()
        # out_data["debug_jet_phi"   ] = event.Jet[event.Jet.selected_eta].phi[debug_mask].to_list()
        # out_data["debug_jet_pu"    ] = event.Jet[event.Jet.selected_eta].pileup[debug_mask].to_list()
        # out_data["debug_jet_jetId" ] = event.Jet[event.Jet.selected_eta].jetId[debug_mask].to_list()
        # out_data["debug_jet_lep"   ] = event.Jet[event.Jet.selected_eta].lepton_cleaned[debug_mask].to_list()
        #
        # for out_k, out_v in out_data.items():
        #     processOutput[out_k] = {}
        #     processOutput[out_k][event.metadata['dataset']] = list(out_v)

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
                           **dict((s, ...) for s in self.histCuts)
                           )

        #
        # To Add
        #

        #    m4j_vs_leadSt_dR = dir.make<TH2F>("m4j_vs_leadSt_dR", (name+"/m4j_vs_leadSt_dR; m_{4j} [GeV]; S_{T} leading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);
        #    m4j_vs_sublSt_dR = dir.make<TH2F>("m4j_vs_sublSt_dR", (name+"/m4j_vs_sublSt_dR; m_{4j} [GeV]; S_{T} subleading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);

        fill += hist.add( "nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")) )
        fill += hist.add( "nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")), )



        if "xbW" in selev.fields:  ### AGE: this should be temporary
            fill += hist.add("xW", (100, 0, 12, ("xW", "xW")))
            #fill += hist.add("delta_xW", (100, -5, 5, ("delta_xW", "delta xW")))
            #fill += hist.add("delta_xW_l", (100, -15, 15, ("delta_xW", "delta xW")))


        #
        # Jets
        #
        fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])
        fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=["deepjet_c"])
        fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=["deepjet_c"])
        fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=["deepjet_c"])

        #
        #  Make quad jet hists
        #
        #fill += LorentzVector.plot_pair( ("v4j", R"$HH_{4b}$"), "v4j", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )
        #fill += QuadJetHists( ("quadJet_selected", "Selected Quad Jet"), "quadJet_selected" )
        fill += QuadJetHists( ("quadJet_min_dr", "Min dR Quad Jet"), "quadJet_min_dr" )

        #
        #  Make Jet Hists
        #
        skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]


        for iJ in range(4):
            fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )

        #
        #  Leptons
        #
        skip_muons = ["charge"] + Muon.skip_detailed_plots
        if not isMC:
            skip_muons += ["genPartFlav"]
        fill += Muon.plot( ("selMuons", "Selected Muons"), "selMuon", skip=skip_muons )

        skip_elecs = ["charge"] + Elec.skip_detailed_plots
        if not isMC:
            skip_elecs += ["genPartFlav"]
        fill += Elec.plot( ("selElecs", "Selected Elecs"), "selElec", skip=skip_elecs )

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
