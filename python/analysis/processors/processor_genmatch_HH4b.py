import time
import gc
import awkward as ak
import numpy as np
import correctionlib
import yaml
import warnings
import uproot
import copy
import hist as hist2
from functools import reduce

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection

from analysis.helpers.cutflow import cutFlow
from analysis.helpers.FriendTreeSchema import FriendTreeSchema

from analysis.helpers.common import init_jet_factory, apply_btag_sf, update_events

from analysis.helpers.selection_basic_4b import (
    apply_event_selection_4b,
    apply_object_selection_4b,
    #apply_object_selection_boosted_4b
)

import logging

from base_class.root import TreeReader, Chunk

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

#### from https://github.com/aebid/HHbbWW_Run3/blob/main/python/genparticles.py#L42
def find_genpart(genpart, pdgid, ancestors):
    """
    Find gen level particles given pdgId (and ancestors ids)

    Parameters:
    genpart (GenPart): NanoAOD GenPart collection.
    pdgid (list): pdgIds for the target particles.
    idmother (list): pdgIds for the ancestors of the target particles.

    Returns:
    NanoAOD GenPart collection
    """

    def check_id(p):
        return np.abs(genpart.pdgId) == p

    pid = reduce(np.logical_or, map(check_id, pdgid))

    if ancestors:
        ancs, ancs_idx = [], []
        for i, mother_id in enumerate(ancestors):
            if i == 0:
                mother_idx = genpart[pid].genPartIdxMother
            else:
                mother_idx = genpart[ancs_idx[i-1]].genPartIdxMother
            ancs.append(np.abs(genpart[mother_idx].pdgId) == mother_id)
            ancs_idx.append(mother_idx)

        decaymatch =  reduce(np.logical_and, ancs)
        return genpart[pid][decaymatch]

    return genpart[pid]


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
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
            "passPreSel",
            "passDiJetMass",
            "SR",
            "SB",
        ]

        self.histCuts = ["passPreSel"]
        self.cutFlowCuts += ["passSvB", "failSvB"]
        self.histCuts += ["passSvB", "failSvB"]


    def process(self, event):

        tstart = time.time()
        fname   = event.metadata['filename']
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        nEvent = len(event)
        weights = Weights(len(event), storeIndividual=True)

        logging.debug(fname)
        logging.debug(f'Process {nEvent} Events')

        #
        # Event selection
        #
        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year], False)

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year] )
        #event = apply_object_selection_boosted_4b( event )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if (isMC or isMixedData or isTTForMixed or isDataForMixed) else event.passHLT ) )
        selections.add( 'passJetMult', event.passJetMult )
        selections.add( "passPreSel", event.passPreSel )
        selections.add( "passFourTag", ( event.passJetMult & event.passPreSel & event.fourTag) )
        #selections.add( 'passBoostedSel', event.passBoostedSel )
        allcuts = [ 'passJetMult' ]

        #
        #  Cut Flows
        #
        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = {
            'numEvents': nEvent,
        }


        event['GenJet', 'selected'] = (event.GenJet.pt >= 40) & (np.abs(event.GenJet.eta) <= 2.4) & (np.abs(event.GenJet.partonFlavour)==5)
        event['selGenJet'] = event.GenJet[event.GenJet.selected]
        event['passFourGenJets'] = ak.num(event.selGenJet) >=4
        selections.add('passFourGenJets', event.passFourGenJets )

        event['bfromH']= find_genpart(event.GenPart, [5], [25])
        event['matchedGenJet'] = event.bfromH.nearest( event.selGenJet, threshold=0.2 )
        event['matchedGenJet'] = event.matchedGenJet[ ak.argsort(event.matchedGenJet.pt, axis=1, ascending=False) ]
        event['m4j'] = ak.where( ak.num(event.matchedGenJet) == 4 ,
                                (event.matchedGenJet[:,0] + event.matchedGenJet[:,1] + event.matchedGenJet[:,2] + event.matchedGenJet[:,3]).mass,
                                -999 )

        event['Jet', 'selected'] = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4)
        event['selJet'] = event.Jet[ event.Jet.selected ]
        event['matchedRecoJet'] = event.bfromH.nearest( event.selJet, threshold=0.2 )
        event['matchedRecoJet'] = event.matchedRecoJet[ ak.argsort(event.matchedRecoJet.pt, axis=1, ascending=False) ]

        hist = { 'hists': {} }
        process_axis = hist2.axis.StrCategory([], name='process', growth=True)
        sel_axis = hist2.axis.StrCategory([], name='selection', growth=True)
        hist['hists']['numGenJets'] = hist2.Hist(
            process_axis, sel_axis,
            hist2.axis.Regular(20, 0, 20, name='n'),
            hist2.storage.Weight()
        )
        hist['hists']['genJet1_pt'] = hist2.Hist(
            process_axis, sel_axis,
            hist2.axis.Regular(200, 0, 2000, name='pt'),
            hist2.storage.Weight()
        )
        hist['hists']['genJet4_pt'] = copy.copy(hist['hists']['genJet1_pt'])

        hist['hists']['matchgenJet1_pt'] = copy.copy(hist['hists']['genJet1_pt'])
        hist['hists']['matchgenJet2_pt'] = copy.copy(hist['hists']['genJet1_pt'])
        hist['hists']['matchgenJet3_pt'] = copy.copy(hist['hists']['genJet1_pt'])
        hist['hists']['matchgenJet4_pt'] = copy.copy(hist['hists']['genJet1_pt'])

        hist['hists']['genJet1_eta'] = hist2.Hist(
            process_axis, sel_axis,
            hist2.axis.Regular(80, -5, 5, name='eta'),
            hist2.storage.Weight()
        )
        hist['hists']['genJet4_eta'] = copy.copy(hist['hists']['genJet1_eta'])
        hist['hists']['matchgenJet1_eta'] = copy.copy(hist['hists']['genJet1_eta'])
        hist['hists']['matchgenJet2_eta'] = copy.copy(hist['hists']['genJet1_eta'])
        hist['hists']['matchgenJet3_eta'] = copy.copy(hist['hists']['genJet1_eta'])
        hist['hists']['matchgenJet4_eta'] = copy.copy(hist['hists']['genJet1_eta'])

        hist['hists']['m4j'] = hist2.Hist(
            process_axis, sel_axis,
            hist2.axis.Regular(200, 0, 2000, name='mass'),
            hist2.storage.Weight()
        )

        hist['hists']['puId'] = hist2.Hist(
            process_axis, sel_axis,
            hist2.axis.Regular( 10, 0, 10, name='puId' ),
            hist2.storage.Double()
        )

        cuts = {
            'noselection' : [],
            'reco_fourtag' : ['passFourTag'],
            'gen_fourtag' : ['passFourGenJets']
        }

        for iname, icut in cuts.items():
            icut = selections.all(*icut)
            hist['hists']['numGenJets'].fill(
                process=dataset,
                selection=iname,
                n=ak.num(event.selGenJet[icut]),
                weight=weights.weight()[icut]
            )

            event['selGenJet'] = ak.pad_none( event.selGenJet, 4, axis=1 )
            hist['hists']['genJet1_pt'].fill(
                process=dataset,
                selection=iname,
                pt=ak.to_numpy(ak.fill_none(event['selGenJet'][icut][:,0].pt, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['genJet4_pt'].fill(
                process=dataset,
                selection=iname,
                pt=ak.to_numpy(ak.fill_none(event['selGenJet'][icut][:,3].pt, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['genJet1_eta'].fill(
                process=dataset,
                selection=iname,
                eta=ak.to_numpy(ak.fill_none(event['selGenJet'][icut][:,0].eta, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['genJet4_eta'].fill(
                process=dataset,
                selection=iname,
                eta=ak.to_numpy(ak.fill_none(event['selGenJet'][icut][:,3].eta, np.nan)),
                weight=weights.weight()[icut]
            )

            hist['hists']['matchgenJet1_pt'].fill(
                process=dataset,
                selection=iname,
                pt=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,0].pt, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['matchgenJet1_eta'].fill(
                process=dataset,
                selection=iname,
                eta=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,0].eta, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['matchgenJet2_pt'].fill(
                process=dataset,
                selection=iname,
                pt=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,1].pt, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['matchgenJet2_eta'].fill(
                process=dataset,
                selection=iname,
                eta=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,1].eta, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['matchgenJet3_pt'].fill(
                process=dataset,
                selection=iname,
                pt=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,2].pt, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['matchgenJet3_eta'].fill(
                process=dataset,
                selection=iname,
                eta=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,2].eta, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['matchgenJet4_pt'].fill(
                process=dataset,
                selection=iname,
                pt=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,3].pt, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['matchgenJet4_eta'].fill(
                process=dataset,
                selection=iname,
                eta=ak.to_numpy(ak.fill_none(event['matchedGenJet'][icut][:,3].eta, np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['m4j'].fill(
                process=dataset,
                selection=iname,
                mass=ak.to_numpy(ak.fill_none(event['m4j'][icut], np.nan)),
                weight=weights.weight()[icut]
            )
            hist['hists']['puId'].fill(
                process=dataset,
                selection=iname,
                puId=ak.to_numpy(ak.fill_none(ak.flatten(event['matchedRecoJet'][icut].puId), np.nan)),
            )

        self._cutFlow = cutFlow(self.cutFlowCuts)
        self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        output = processOutput | hist

        return output

        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f"{chunk}{nEvent/elapsed:,.0f} events/s")


    def postprocess(self, accumulator):
        return accumulator
