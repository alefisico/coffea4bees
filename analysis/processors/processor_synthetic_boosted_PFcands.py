import time
import awkward as ak
import numpy as np
import yaml
import warnings
from collections import OrderedDict
from src.hist import Template

from coffea.nanoevents import NanoAODSchema
from coffea import processor
from coffea.analysis_tools import PackedSelection
import hist
from python.analysis.helpers.cutflow import cutflow_4b

from src.physics.event_selection import apply_event_selection
from src.hist_tools import Collection, Fill
from coffea4bees.jet_clustering.clustering_hist_templates import ClusterHistsBoosted
from src.hist_tools.object import LorentzVector, Jet, PFCand

from coffea4bees.jet_clustering.declustering import compute_decluster_variables
from coffea4bees.jet_clustering.clustering import comb_jet_flavor


import logging
import vector

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            corrections_metadata: dict = None,
            **kwargs
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata

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


        selFatJet = event.FatJet
        #selFatJet = selFatJet[selFatJet.particleNetMD_Xbb > 0.8]
        selFatJet = selFatJet[selFatJet.subJetIdx1 >= 0]
        selFatJet = selFatJet[selFatJet.subJetIdx2 >= 0]

        selFatJet = event.FatJet[event.FatJet.pt > 300]

        selFatJet = selFatJet[(selFatJet.subjets [:, :, 0] + selFatJet.subjets [:, :, 1]).pt   > 300]
        selFatJet = selFatJet[(selFatJet.subjets [:, :, 0] + selFatJet.subjets [:, :, 1]).mass > 50]

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

        event["weight"] = 1.0


        #
        # Do the cutflow
        #
        sel_dict = OrderedDict({
            'all'               : selections.require(lumimask=True),
            'passNoiseFilter'   : selections.require(lumimask=True, passNoiseFilter=True),
            'passHLT'           : selections.require(lumimask=True, passNoiseFilter=True, passHLT=True),
            'passNFatJets'      : selections.require(lumimask=True, passNoiseFilter=True, passHLT=True, passNFatJets=True),
        })
        #sel_dict['passJetMult'] = selections.all(*allcuts)

        self.cutFlow = cutflow_4b()
        for cut, sel in sel_dict.items():
            self.cutFlow.fill( cut, event[sel], allTag=True )


        list_of_cuts = [ "lumimask", "passNoiseFilter", "passHLT", "passNFatJets" ]
        #list_of_cuts = [ "passNFatJets" ]
        analysis_selections = selections.all(*list_of_cuts)
        selev = event[analysis_selections]


        #
        #  Create the subjets
        #
        sorted_sub_jets = selev.selFatJet.subjets
        sorted_sub_jets = sorted_sub_jets[ak.argsort(sorted_sub_jets.pt, axis=2, ascending=False)]

        if "tau1" not in sorted_sub_jets.fields:
            # If the subjets do not have tau1, tau2, tau3, we need to add them
            sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau1")
            sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau2")
            sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau3")


        PFCandFatJet0_mask = selev.FatJetPFCands.jetIdx == ak.local_index(selev.FatJet, axis=1)[:,0]
        PFCandFatJet1_mask = selev.FatJetPFCands.jetIdx == ak.local_index(selev.FatJet, axis=1)[:,1]


        PFCandIndex_FatJet0 = selev.FatJetPFCands[PFCandFatJet0_mask].pFCandsIdx
        PFCandIndex_FatJet1 = selev.FatJetPFCands[PFCandFatJet1_mask].pFCandsIdx

        # print(f" nPFCands for FatJet0 {ak.num(PFCandIndex_FatJet0)}\n")
        # print(f" nPFCands for FatJet1 {ak.num(PFCandIndex_FatJet1)}\n")
        #
        # print(f" PFCands for FatJet0 {selev.PFCands[PFCandIndex_FatJet0].pdgId.tolist()}\n")
        # print(f" PFCands for FatJet1 {selev.PFCands[PFCandIndex_FatJet1].pdgId.tolist()}\n")


        PFCands_perFatJet = ak.Array([[a, b] for a, b in zip(selev.PFCands[PFCandIndex_FatJet0], selev.PFCands[PFCandIndex_FatJet1])])

        fatjets = ak.zip({"p"  : sorted_sub_jets[:, :, 0] + sorted_sub_jets[:, :, 1],
                          "i0" : sorted_sub_jets[:, :, 0],
                          "i1" : sorted_sub_jets[:, :, 1],
                          "PFCands": PFCands_perFatJet,
                          },
                         depth_limit=1,
                         )

        # Calculate di-jet variables
        fatjets["p", "st"]   = fatjets.i0.pt + fatjets.i1.pt
        fatjets["p", "dr"]   = fatjets.i0.delta_r(fatjets.i1)
        fatjets["p", "dphi"] = fatjets.i0.delta_phi(fatjets.i1)


        # Define the tau21 variable for each subjet
        fatjets["i0", "tau21"] = fatjets.i0.tau2 / (fatjets.i0.tau1 + 0.001)
        fatjets["i1", "tau21"] = fatjets.i1.tau2 / (fatjets.i1.tau1 + 0.001)

        fatjets["i0", "tau32"] = fatjets.i0.tau3 / (fatjets.i0.tau2 + 0.001)
        fatjets["i1", "tau32"] = fatjets.i1.tau3 / (fatjets.i1.tau2 + 0.001)

        # Dummy
        fatjets["i0", "btag_string"] = "0.001"
        fatjets["i1", "btag_string"] = "0.001"


        # Define the jet flavor for each subjet
        fatjets["i0", "jet_flavor"] = ak.where(fatjets.i0.tau21 > 0.5, "b", "bj")
        fatjets["i1", "jet_flavor"] = ak.where(fatjets.i1.tau21 > 0.5, "b", "bj")

        # Swap if i1 more complex than i0
        i1_more_complex = (fatjets.i1.jet_flavor == "bj") & (fatjets.i0.jet_flavor == "b")

        fatjets["iA"] = ak.where(i1_more_complex, fatjets.i1, fatjets.i0)
        fatjets["iB"] = ak.where(i1_more_complex, fatjets.i0, fatjets.i1)

        #
        #  Set the jet flavor for the combined fat jet
        #
        C = [comb_jet_flavor(a, b) for a, b in zip(ak.flatten(fatjets.iA.jet_flavor), ak.flatten(fatjets.iB.jet_flavor))]
        fatjet_flavor_flat = np.array(C)
        selev["selFatJet", "jet_flavor"] = ak.unflatten(fatjet_flavor_flat, ak.num(selev.selFatJet))


        #
        #  Printouts to understand the structure of the PFCands
        #
        # print(f"Number of FatJet PFCands: {ak.num(selev.FatJetPFCands)}\n")
        # print(f"   event fields: {selev.fields}\n")
        # print(f"   FatJet PFCands pt 0: {selev.FatJetPFCands.pt[0].tolist()}\n")
        # print(f"   FatJet PFCands pfCandIdx 0: {selev.FatJetPFCands.pFCandsIdx[0].tolist()}\n")
        # print(f"   FatJet PFCands jetIdx 0: {selev.FatJetPFCands.jetIdx[0].tolist()}\n")
        # print(f"FatJet PFCands fields: {selev.FatJetPFCands.fields}\n")
        # print(f"FatJet Fat Jet fields: {selev.FatJet.fields}\n")
        # print(f"FatJet FatJet dir: {dir(selev.FatJet)}\n")
        # print(f"PFCand fields: {selev.PFCands.fields}\n")

        #print(f" nPFCandsR {ak.num(selev.PFCands[result])}\n")
        #print(f" PFCandsR pdgId {selev.PFCands[result].pdgId.tolist()}\n")


        #print(type(selev.selFatJet),"\n")
        #selev.PFCands[PFCandIndex_FatJet0]

        #fatjets["PFCand1"] = selev.PFCands[PFCandIndex_FatJet1]
        selev["selFatJet_PFCand0"] = selev.PFCands[PFCandIndex_FatJet0]


        #
        #  Add the fatjets to the event
        #
        selev["fatjets"] = fatjets


        #print(f" PFCands for FatJet1 {selev.FatJetPFCands[PFCandFatJet1_mask].pFCandsIdx.tolist()}\n")
        #print(f"   FatJet PFCands jetIdx 0: {selev.FatJetPFCands.jetIdx[0].tolist()}\n")

        #print(f"Number of selected Fat Jets: {ak.num(selev.selFatJet)}")
        #print(f" Any passNFatJets: {ak.any(selev.passNFatJets)}")
        #print(f" Any passHLT: {ak.any(selev.passHLT)}")
        #print(f" FatJet pt: {selev.selFatJet.pt}")
        #
        #print(f" nFatjets: {ak.num(selev.selFatJet.fatjets, axis=2)}")
        #print(f" subjet pt: {selev.selFatJet.pt[0:10]}")


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


        #
        # Jets
        #
        selev["selFatJet", "btagScore"] = selev.selFatJet.particleNet_XbbVsQCD
        fill += Jet.plot(("fatJets", "Selected Fat Jets"),        "selFatJet",           skip=["deepjet_c"], bins={"pt": (50, 0, 1000)})



        #
        #  Class to plot the Fat Jets
        #
        class FatJetHists(Template):

            p  = LorentzVector.plot_pair(('...', R'Fat Jet'), 'p',  skip=['n'], bins={"mass": (100, 40, 300),
                                                                                      "pt": (60, 250, 1000),
                                                                                      "dr": (50, 0, 1.2),
                                                                                      "dphi": (50, -1.5, 1.5)
                                                                                      })

            i0 = Jet.plot(('...', R'subjet 0'), 'i0',     skip=['deepjet_c','n'], bins={"mass": (100, 0, 200), "pt": (50, 100, 1000)})
            i1 = Jet.plot(('...', R'subjet 1'), 'i1',     skip=['deepjet_c','n'], bins={"mass": (50, 0, 100)})

            pf = PFCand.plot(('...', R'PFCands in selected fat jet'), "PFCands",   skip=[],
                             bins={"pt":   (50, 0, 10), "mass": (50, 0, 0.2)}
                             )




        fill += FatJetHists(('fatJets', R''), 'fatjets')
        #trkPt
        #print("fields",selev["selFatJet_PFCand0"].fields,"\n")

        #fill += hist.add( "PFCand0.pt_l",  (100, 0, 100, ("selFatJet_PFCand0.trkPt",   'trk pt')))
        #fill += hist.add( "PFCand0.pt",    (100, 0, 10,  ("selFatJet_PFCand0.trkPt",   'trk pt')))

#        for _s_type in cleaned_splitting_name:

        #
        # fill histograms
        #
        fill(selev, hist)

        processOutput = {}

        output = hist.output | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
