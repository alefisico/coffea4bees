import os
import time
import gc
import argparse
import sys
from copy import deepcopy
import awkward as ak
import numpy as np
import uproot
import correctionlib
import correctionlib._core as core
import yaml
import warnings
import torch
import torch.nn.functional as F

from analysis.helpers.networks import HCREnsemble
from analysis.helpers.topCandReconstruction import find_tops, dumpTopCandidateTestVectors, buildTop

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector
from coffea import processor

from base_class.hist import Collection, Fill
from base_class.physics.object import LorentzVector, Jet, Muon, Elec
from analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists, WCandHists, TopCandHists

from analysis.helpers.cutflow import cutFlow
from analysis.helpers.FriendTreeSchema import FriendTreeSchema
from analysis.helpers.correctionFunctions import btagVariations
from analysis.helpers.correctionFunctions import btagSF_norm as btagSF_norm_file
from functools import partial
from multiprocessing import Pool

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
# print(torch.__config__.parallel_info())

from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.common import apply_btag_sf
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b
import logging


#
# Setup
#
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")
ak.behavior.update(vector.behavior)

def setSvBVars(SvBName, event):
    largest_name = np.array(['None', 'ZZ', 'ZH', 'HH'])

    event[SvBName, 'passMinPs'] = (getattr(event, SvBName).pzz > 0.01) | (getattr(event, SvBName).pzh > 0.01) | (getattr(event, SvBName).phh > 0.01)
    event[SvBName, 'zz'] = (getattr(event, SvBName).pzz >  getattr(event, SvBName).pzh) & (getattr(event, SvBName).pzz >  getattr(event, SvBName).phh)
    event[SvBName, 'zh'] = (getattr(event, SvBName).pzh >  getattr(event, SvBName).pzz) & (getattr(event, SvBName).pzh >  getattr(event, SvBName).phh)
    event[SvBName, 'hh'] = (getattr(event, SvBName).phh >= getattr(event, SvBName).pzz) & (getattr(event, SvBName).phh >= getattr(event, SvBName).pzh)
    event[SvBName, 'largest'] = largest_name[ getattr(event, SvBName).passMinPs*(1*getattr(event, SvBName).zz + 2*getattr(event, SvBName).zh + 3*getattr(event, SvBName).hh) ]

    #
    #  Set ps_{bb}
    #
    this_ps_zz = np.full(len(event), -1, dtype=float)
    this_ps_zz[getattr(event, SvBName).zz] = getattr(event, SvBName).pzz[getattr(event, SvBName).zz]
    event[SvBName, 'ps_zz'] = this_ps_zz

    this_ps_zh = np.full(len(event), -1, dtype=float)
    this_ps_zh[getattr(event, SvBName).zh] = getattr(event, SvBName).pzh[getattr(event, SvBName).zh]
    event[SvBName, 'ps_zh'] = this_ps_zh

    this_ps_hh = np.full(len(event), -1, dtype=float)
    this_ps_hh[getattr(event, SvBName).hh] = getattr(event, SvBName).phh[getattr(event, SvBName).hh]
    event[SvBName, 'ps_hh'] = this_ps_hh


class analysis(processor.ProcessorABC):
    def __init__(self, *, JCM = None, addbtagVariations=None, SvB=None, SvB_MA=None, threeTag = True, apply_trigWeight = True, apply_btagSF = True, apply_FvT = True, run_SvB = True, run_topreco = True, corrections_metadata='analysis/metadata/corrections.yml', make_classifier_input: str = None):
        logging.debug('\nInitialize Analysis Processor')
        self.blind = False
        print('Initialize Analysis Processor')
        self.JCM = jetCombinatoricModel(JCM) if JCM else None
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.btagVar = btagVariations(systematics=addbtagVariations)  #### AGE: these two need to be review later
        self.run_SvB = run_SvB
        self.run_topreco = run_topreco  #### AGE: this is temporary topreco is memory consuming (needs fix)
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, 'r'))
        self.cutFlowCuts = ["all", "passHLT", "passNoiseFilter", "passJetMult", "passJetMult_btagSF", "passPreSel", "passDiJetMass", 'SR', 'SB']
        self.histCuts = ['passPreSel']
        if self.run_SvB:
            self.cutFlowCuts += ['passSvB', 'failSvB']
            self.histCuts += ['passSvB', 'failSvB']
        self.make_classifier_input = make_classifier_input

    def process(self, event):
        tstart = time.time()

        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        era     = event.metadata.get('era', '')
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        nEvent = len(event)

        #
        #  Cut Flows
        #
        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = nEvent

        self._cutFlow = cutFlow(self.cutFlowCuts)

        logging.debug(fname)
        logging.debug(f'{chunk}Process {nEvent} Events')

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split('/')[-1], '')
        if self.apply_FvT:
            event['FvT']    = NanoEventsFactory.from_root(f'{path}{"FvT_3bDvTMix4bDvT_v0_newSB.root" if "mix" in dataset else "FvT.root"}',
                                                          entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().FvT
            event['FvT', 'frac_err'] = event['FvT'].std / event['FvT'].FvT
            if not ak.all(event.FvT.event == event.event):
                logging.error('ERROR: FvT events do not match events ttree')
                return

        if self.run_SvB:
            if (self.classifier_SvB is not None) | (self.classifier_SvB_MA is not None):
                self.compute_SvB(selev)  ### this computes both
            else:
                event['SvB']    = NanoEventsFactory.from_root(f'{path}{"SvB_newSBDef.root" if "mix" in dataset else "SvB.root"}',
                                                          entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().SvB
                if not ak.all(event.SvB.event == event.event):
                    logging.error('ERROR: SvB events do not match events ttree')
                    return

                event['SvB_MA'] = NanoEventsFactory.from_root(f'{path}{"SvB_MA_newSBDef.root" if "mix" in dataset else "SvB_MA.root"}',
                                                          entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().SvB_MA
                if not ak.all(event.SvB_MA.event == event.event):
                    logging.error('ERROR: SvB_MA events do not match events ttree')
                    return

                # defining SvB for different SR
                setSvBVars("SvB",    event)
                setSvBVars("SvB_MA", event)

        #
        # general event weights
        #
        if isMC:
            # genWeight
            genEventSumw = event.metadata['genEventSumw']
            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)
            logging.debug(f"event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw) = {event.genWeight[0]} * ({lumi} * {xs} * {kFactor} / {genEventSumw}) = {event.weight[0]}\n")

            # trigger Weight (to be updated)
            if self.apply_trigWeight: event['weight'] = event.weight * event.trigWeight.Data

            # puWeight
            puWeight = list(correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['PU']).values())[0]
            for var in ['nominal', 'up', 'down']:
                event[f'PU_weight_{var}'] = puWeight.evaluate(event.Pileup.nTrueInt.to_numpy(), var)
            event['weight'] = event.weight * event.PU_weight_nominal

            # L1 prefiring weight
            if ('L1PreFiringWeight' in event.fields):   #### AGE: this should be temprorary (field exists in UL)
                event['weight'] = event.weight * event.L1PreFiringWeight.Nom
        else:
            event['weight'] = 1

        logging.debug(f"event['weight'] = {event.weight}")

        #
        # Event selection (function only adds flags, not remove events)
        #
        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )

        self._cutFlow.fill("all",  event[event.lumimask], allTag=True)
        self._cutFlow.fill("passNoiseFilter",  event[ event.lumimask & event.passNoiseFilter], allTag=True)
        self._cutFlow.fill("passHLT",  event[ event.lumimask & event.passNoiseFilter & event.passHLT], allTag=True)

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year]  )
        basic_selection = event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult
        self._cutFlow.fill("passJetMult", event[basic_selection], allTag=True)

        #
        # Filtering object and event selection
        #
        selev = event[basic_selection]

        #
        # Calculate and apply btag scale factors
        #
        if isMC and self.apply_btagSF:
            btagSF = correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['btagSF'])['deepJet_shape']
            selev['weight'] = apply_btag_sf(selev, selev.selJet,
                                            correction_file=self.corrections_metadata[year]['btagSF'],
                                            btag_var=self.btagVar,
                                            btagSF_norm=btagSF_norm_file(dataset),
                                            weight=selev.weight )

            self._cutFlow.fill("passJetMult_btagSF",  selev, allTag=True)

        #
        # Preselection: keep only three or four tag events
        #
        selev = selev[selev.passPreSel]

        #
        #  Calculate hT
        #
        selev['hT']          = ak.sum(selev.Jet[selev.Jet.selected_loose].pt, axis=1)
        selev['hT_selected'] = ak.sum(selev.Jet[selev.Jet.selected      ].pt, axis=1)

        #
        # Build and select boson candidate jets with bRegCorr applied
        #
        sorted_idx = ak.argsort(selev.Jet.btagDeepFlavB * selev.Jet.selected, axis=1, ascending=False)
        canJet_idx    = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]
        canJet = selev.Jet[canJet_idx]

        # apply bJES to canJets
        canJet = canJet * canJet.bRegCorr
        canJet['bRegCorr'] = selev.Jet.bRegCorr[canJet_idx]
        canJet['btagDeepFlavB'] = selev.Jet.btagDeepFlavB[canJet_idx]
        canJet['puId'] = selev.Jet.puId[canJet_idx]
        canJet['jetId'] = selev.Jet.puId[canJet_idx]
        if isMC:
            canJet['hadronFlavour'] = selev.Jet.hadronFlavour[canJet_idx]
        canJet['calibration'] = selev.Jet.calibration[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        selev['canJet'] = canJet

        #
        #  Should be a better way to do this...
        #
        selev['canJet0'] = canJet[:, 0]
        selev['canJet1'] = canJet[:, 1]
        selev['canJet2'] = canJet[:, 2]
        selev['canJet3'] = canJet[:, 3]

        selev['v4j'] = canJet.sum(axis=1)
        # selev['v4j', 'n'] = 1
        # print(selev.v4j.n)
        # selev['Jet', 'canJet'] = False
        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        notCanJet['isSelJet'] = 1 * ((notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4))     # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev['notCanJet_coffea'] = notCanJet
        selev['nNotCanJet'] = ak.num(selev.notCanJet_coffea)

        #
        # calculate pseudoTagWeight for threeTag events
        #
        selev['weight_noJCM_noFvT'] = selev.weight
        if self.JCM:
            selev['Jet_untagged_loose'] = selev.Jet[selev.Jet.selected & ~selev.Jet.tagged_loose]
            nJet_pseudotagged = np.zeros(len(selev), dtype=int)
            pseudoTagWeight = np.ones(len(selev))
            pseudoTagWeight[selev.threeTag], nJet_pseudotagged[selev.threeTag] = self.JCM(selev[selev.threeTag]['Jet_untagged_loose'])
            selev['nJet_pseudotagged'] = nJet_pseudotagged
            selev['pseudoTagWeight'] = pseudoTagWeight

            #
            # apply pseudoTagWeight and FvT to threeTag events
            #
            weight_noFvT = np.array(selev.weight.to_numpy(), dtype=float)
            weight_noFvT[selev.threeTag] = selev.weight[selev.threeTag] * selev.pseudoTagWeight[selev.threeTag]
            selev['weight_noFvT'] = weight_noFvT

            if self.apply_FvT:
                weight = np.array(selev.weight.to_numpy(), dtype=float)
                weight[selev.threeTag] = selev.weight[selev.threeTag] * pseudoTagWeight[selev.threeTag] * selev.FvT.FvT[selev.threeTag]
                selev['weight'] = weight
            else:
                selev['weight'] = weight_noFvT

        #
        # Build diJets, indexed by diJet[event,pairing,0/1]
        #
        canJet = selev['canJet']
        pairing = [([0, 2], [0, 1], [0, 1]),
                   ([1, 3], [2, 3], [3, 2])]
        diJet       = canJet[:, pairing[0]]     +   canJet[:, pairing[1]]
        diJet['st'] = canJet[:, pairing[0]].pt  +   canJet[:, pairing[1]].pt
        diJet['dr'] = canJet[:, pairing[0]].delta_r(canJet[:, pairing[1]])
        diJet['dphi'] = canJet[:, pairing[0]].delta_phi(canJet[:, pairing[1]])
        diJet['lead'] = canJet[:, pairing[0]]
        diJet['subl'] = canJet[:, pairing[1]]
        # Sort diJets within views to be lead st, subl st
        diJet   = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
        diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]
        # Now indexed by diJet[event,pairing,lead/subl st]

        # Compute diJetMass cut with independent min/max for lead/subl
        minDiJetMass = np.array([[[ 52,  50]]])
        maxDiJetMass = np.array([[[180, 173]]])
        diJet['passDiJetMass'] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

        # Compute MDRs
        min_m4j_scale = np.array([[ 360, 235]])
        min_dr_offset = np.array([[-0.5, 0.0]])
        max_m4j_scale = np.array([[ 650, 650]])
        max_dr_offset = np.array([[ 0.5, 0.7]])
        max_dr        = np.array([[ 1.5, 1.5]])
        m4j = np.repeat(np.reshape(np.array(selev['v4j'].mass), (-1, 1, 1)), 2, axis=2)
        diJet['passMDR'] = (min_m4j_scale / m4j + min_dr_offset < diJet.dr) & (diJet.dr < np.maximum(max_m4j_scale / m4j + max_dr_offset, max_dr))

        #
        # Compute consistency of diJet masses with boson masses
        #
        mZ =  91.0
        mH = 125.0
        st_bias = np.array([[[1.02, 0.98]]])
        cZ = mZ * st_bias
        cH = mH * st_bias

        diJet['xZ'] = (diJet.mass - cZ) / (0.1 * diJet.mass)
        diJet['xH'] = (diJet.mass - cH) / (0.1 * diJet.mass)

        #
        # Build quadJets
        #
        seeds = np.array(event.event)[[0, -1]].view(np.ulonglong)
        randomstate = np.random.Generator(np.random.PCG64(seeds))
        quadJet = ak.zip({'lead': diJet[:, :, 0],
                          'subl': diJet[:, :, 1],
                          'close': diJetDr[:, :, 0],
                          'other': diJetDr[:, :, 1],
                          'passDiJetMass': ak.all(diJet.passDiJetMass, axis=2),
                          'random': randomstate.uniform(low=0.1, high=0.9, size=(diJet.__len__(), 3))})

        quadJet['dr']   = quadJet['lead'].delta_r(quadJet['subl'])
        quadJet['dphi'] = quadJet['lead'].delta_phi(quadJet['subl'])
        quadJet['deta'] = quadJet['lead'].eta - quadJet['subl'].eta

        if self.apply_FvT:
            quadJet['FvT_q_score'] = np.concatenate((np.reshape(np.array(selev.FvT.q_1234), (-1, 1)),
                                                     np.reshape(np.array(selev.FvT.q_1324), (-1, 1)),
                                                     np.reshape(np.array(selev.FvT.q_1423), (-1, 1))), axis=1)

        if self.run_SvB:
            quadJet['SvB_q_score'] = np.concatenate((np.reshape(np.array(selev.SvB.q_1234), (-1, 1)),
                                                     np.reshape(np.array(selev.SvB.q_1324), (-1, 1)),
                                                     np.reshape(np.array(selev.SvB.q_1423), (-1, 1))), axis=1)

            quadJet['SvB_MA_q_score'] = np.concatenate((np.reshape(np.array(selev.SvB_MA.q_1234), (-1, 1)),
                                                        np.reshape(np.array(selev.SvB_MA.q_1324), (-1, 1)),
                                                        np.reshape(np.array(selev.SvB_MA.q_1423), (-1, 1))), axis=1)
        #
        # Compute Signal Regions
        #
        quadJet['xZZ'] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
        quadJet['xHH'] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
        quadJet['xZH'] = np.sqrt(np.minimum(quadJet.lead.xH**2 + quadJet.subl.xZ**2,
                                            quadJet.lead.xZ**2 + quadJet.subl.xH**2))

        max_xZZ = 2.6
        max_xZH = 1.9
        max_xHH = 1.9
        quadJet['ZZSR'] = quadJet.xZZ < max_xZZ
        quadJet['ZHSR'] = quadJet.xZH < max_xZH
        quadJet['HHSR'] = quadJet.xHH < max_xHH
        quadJet['SR'] = quadJet.ZZSR | quadJet.ZHSR | quadJet.HHSR
        quadJet['SB'] = quadJet.passDiJetMass & ~quadJet.SR

        #
        #  Build the close dR and other quadjets
        #    (There is Probably a better way to do this ...
        #
        arg_min_close_dr = np.argmin(quadJet.close.dr, axis=1)
        arg_min_close_dr = arg_min_close_dr.to_numpy()
        selev['quadJet_min_dr'] = quadJet[np.array(range(len(quadJet))), arg_min_close_dr]

        #
        # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
        #
        quadJet['rank'] = 10 * quadJet.passDiJetMass + quadJet.lead.passMDR + quadJet.subl.passMDR + quadJet.random
        quadJet['selected'] = quadJet.rank == np.max(quadJet.rank, axis=1)

        selev['diJet'] = diJet
        selev['quadJet'] = quadJet
        selev['quadJet_selected'] = quadJet[quadJet.selected][:, 0]
        selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)

        selev['region'] = selev['quadJet_selected'].SR * 0b10 + selev['quadJet_selected'].SB * 0b01
        if self.run_SvB:
            selev['passSvB'] = (selev['SvB_MA'].ps > 0.80)
            selev['failSvB'] = (selev['SvB_MA'].ps < 0.05)

        #
        #  Build the top Candiates
        #
        # dumpTopCandidateTestVectors(selev, logging, chunk, 15)

        if self.run_topreco:
            # sort the jets by btagging
            selev.selJet  = selev.selJet[ak.argsort(selev.selJet.btagDeepFlavB, axis=1, ascending=False)]
            top_cands     = find_tops(selev.selJet)
            rec_top_cands = buildTop(selev.selJet, top_cands)

            selev["top_cand"] = rec_top_cands[:, 0]

            selev["xbW_reco"] = selev.top_cand.xbW
            selev["xW_reco"]  = selev.top_cand.xW

            if 'xbW' in selev.fields:  #### AGE: this should be temporary
                selev["delta_xbW"] = selev.xbW - selev.xbW_reco
                selev["delta_xW"] = selev.xW - selev.xW_reco

        #
        # Blind data in fourTag SR
        #
        if not (isMC or 'mixed' in dataset) and self.blind:
            selev = selev[~(selev['quadJet_selected'].SR & selev.fourTag)]

        #
        # Hists
        #
        fill = Fill(process=processName, year=year, weight='weight')

        hist = Collection(process = [processName],
                          year    = [year],
                          tag     = [3, 4, 0],    # 3 / 4/ Other
                          region  = [2, 1, 0],    # SR / SB / Other
                          **dict((s, ...) for s in self.histCuts))

        #
        # To Add
        #

        #    nIsoMed25Muons = dir.make<TH1F>("nIsoMed25Muons", (name+"/nIsoMed25Muons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
        #    nIsoMed40Muons = dir.make<TH1F>("nIsoMed40Muons", (name+"/nIsoMed40Muons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
        #    muons_isoMed25  = new muonHists(name+"/muon_isoMed25", fs, "iso Medium 25 Muons");
        #    muons_isoMed40  = new muonHists(name+"/muon_isoMed40", fs, "iso Medium 40 Muons");

        #    nIsoMed25Elecs = dir.make<TH1F>("nIsoMed25Elecs", (name+"/nIsoMed25Elecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
        #    nIsoMed40Elecs = dir.make<TH1F>("nIsoMed40Elecs", (name+"/nIsoMed40Elecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
        #    elecs_isoMed25  = new elecHists(name+"/elec_isoMed25", fs, "iso Medium 25 Elecs");
        #    elecs_isoMed40  = new elecHists(name+"/elec_isoMed40", fs, "iso Medium 40 Elecs");
        #

        #    m4j_vs_leadSt_dR = dir.make<TH2F>("m4j_vs_leadSt_dR", (name+"/m4j_vs_leadSt_dR; m_{4j} [GeV]; S_{T} leading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);
        #    m4j_vs_sublSt_dR = dir.make<TH2F>("m4j_vs_sublSt_dR", (name+"/m4j_vs_sublSt_dR; m_{4j} [GeV]; S_{T} subleading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);

        fill += hist.add('nPVs',     (101, -0.5, 100.5, ('PV.npvs',     'Number of Primary Vertices')))
        fill += hist.add('nPVsGood', (101, -0.5, 100.5, ('PV.npvsGood', 'Number of Good Primary Vertices')))

        fill += hist.add('hT',          (100,  0,   1000,  ('hT',          'H_{T} [GeV}')))
        fill += hist.add('hT_selected', (100,  0,   1000,  ('hT_selected', 'H_{T} (selected jets) [GeV}')))

        if ('xbW' in selev.fields) and (self.run_topreco):  ### AGE: this should be temporary

            fill += hist.add('xW',          (100, 0, 12,   ('xW',       'xW')))
            fill += hist.add('delta_xW',    (100, -5, 5,   ('delta_xW', 'delta xW')))
            fill += hist.add('delta_xW_l',  (100, -15, 15, ('delta_xW', 'delta xW')))

            fill += hist.add('xbW',         (100, 0, 12,   ('xbW',      'xbW')))
            fill += hist.add('delta_xbW',   (100, -5, 5,   ('delta_xbW','delta xbW')))
            fill += hist.add('delta_xbW_l', (100, -15, 15, ('delta_xbW','delta xbW')))

        #
        #  Make quad jet hists
        #
        fill += LorentzVector.plot_pair(('v4j', R'$HH_{4b}$'), 'v4j', skip=['n', 'dr', 'dphi', 'st'], bins={'mass': (120, 0, 1200)})
        fill += QuadJetHists(('quadJet_selected', 'Selected Quad Jet'), 'quadJet_selected')
        fill += QuadJetHists(('quadJet_min_dr',   'Min dR Quad Jet'),   'quadJet_min_dr')

        #
        #  Make classifier hists
        #
        if self.apply_FvT:
            fill += FvTHists(('FvT', 'FvT Classifier'), 'FvT')
            fill += hist.add('quadJet_selected_FvT_score', (100, 0, 1, ("quadJet_selected.FvT_q_score", 'Selected Quad Jet Diboson FvT q score')))
            fill += hist.add('quadJet_min_FvT_score', (100, 0, 1, ("quadJet_min_dr.FvT_q_score", 'Min dR Quad Jet Diboson FvT q score')))
            if self.JCM: fill += hist.add('FvT_noFvT', (100, 0, 5, ('FvT.FvT', 'FvT reweight')), weight="weight_noFvT")

        if self.run_SvB:
            fill += SvBHists(('SvB', 'SvB Classifier'), 'SvB')
            fill += SvBHists(('SvB_MA', 'SvB MA Classifier'), 'SvB_MA')
            fill += hist.add('quadJet_selected_SvB_q_score', (100, 0, 1, ("quadJet_selected.SvB_q_score", 'Selected Quad Jet Diboson SvB q score')))
            fill += hist.add('quadJet_min_SvB_MA_q_score', (100, 0, 1, ("quadJet_min_dr.SvB_MA_q_score", 'Min dR Quad Jet Diboson SvB MA q score')))

        #
        # Jets
        #
        fill += Jet.plot(('selJets', 'Selected Jets'),        'selJet',           skip=['deepjet_c'])
        fill += Jet.plot(('canJets', 'Higgs Candidate Jets'), 'canJet',           skip=['deepjet_c'])
        fill += Jet.plot(('othJets', 'Other Jets'),           'notCanJet_coffea', skip=['deepjet_c'])
        fill += Jet.plot(('tagJets', 'Tag Jets'),             'tagJet',           skip=['deepjet_c'])

        skip_all_but_n = ['deepjet_b', 'energy', 'eta', 'id_jet', 'id_pileup', 'mass', 'phi', 'pt', 'pz', 'deepjet_c']
        fill += Jet.plot(('selJets_noJCM', 'Selected Jets'), 'selJet', weight="weight_noJCM_noFvT", skip=skip_all_but_n)
        fill += Jet.plot(('tagJets_noJCM', 'Tag Jets'),      'tagJet', weight="weight_noJCM_noFvT", skip=skip_all_but_n)
        fill += Jet.plot(('tagJets_loose_noJCM', 'Loose Tag Jets'),   'tagJet_loose', weight="weight_noJCM_noFvT", skip=skip_all_but_n)

        for iJ in range(4):
            fill += Jet.plot((f'canJet{iJ}', f'Higgs Candidate Jets {iJ}'), f'canJet{iJ}', skip=['n', 'deepjet_c'])

        #
        #  Leptons
        #
        skip_muons = ['charge'] + Muon.skip_detailed_plots
        if not isMC: skip_muons += ['genPartFlav']
        fill += Muon.plot(('selMuons', 'Selected Muons'),        'selMuon', skip=skip_muons)

        skip_elecs = ['charge'] + Elec.skip_detailed_plots
        if not isMC: skip_elecs += ['genPartFlav']
        fill += Elec.plot(('selElecs', 'Selected Elecs'),        'selElec', skip=skip_elecs)

        #
        # Top Candidates
        #
        #if self.run_topreco:
        #    fill += TopCandHists(('top_cand', 'Top Candidate'), 'top_cand')

        #
        # fill histograms
        #
        # fill.cache(selev)
        fill(selev)

        #
        # CutFlow
        #
        self._cutFlow.fill("passPreSel", selev)
        self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
        if self.run_SvB:
            self._cutFlow.fill("SR",            selev[(selev.passDiJetMass & selev['quadJet_selected'].SR)])
            self._cutFlow.fill("SB",            selev[(selev.passDiJetMass & selev['quadJet_selected'].SB)])
            self._cutFlow.fill("passSvB",       selev[selev.passSvB])
            self._cutFlow.fill("failSvB",       selev[selev.failSvB])

        garbage = gc.collect()
        # print('Garbage:',garbage)

        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f'{chunk}{nEvent/elapsed:,.0f} events/s')

        self._cutFlow.addOutput(processOutput, event.metadata['dataset'])

        friends = {}
        if self.make_classifier_input is not None:
            from ..helpers.classifier.HCR import build_input_friend
            friends['friends'] = build_input_friend(
                selev,
                self.make_classifier_input,
                'HCR_input',
                basic_selection,
                selev.passPreSel,
            )

        return hist.output | processOutput | friends

    def compute_SvB(self, event):
        n = len(event)

        j = torch.zeros(n, 4, 4)
        j[:, 0, :] = torch.tensor(event.canJet.pt  )
        j[:, 1, :] = torch.tensor(event.canJet.eta )
        j[:, 2, :] = torch.tensor(event.canJet.phi )
        j[:, 3, :] = torch.tensor(event.canJet.mass)

        o = torch.zeros(n, 5, 8)
        o[:, 0, :] = torch.tensor(ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.pt,       target=8, clip=True)),  0))
        o[:, 1, :] = torch.tensor(ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.eta,      target=8, clip=True)),  0))
        o[:, 2, :] = torch.tensor(ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.phi,      target=8, clip=True)),  0))
        o[:, 3, :] = torch.tensor(ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.mass,     target=8, clip=True)),  0))
        o[:, 4, :] = torch.tensor(ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.isSelJet, target=8, clip=True)), -1))

        a = torch.zeros(n, 4)
        a[:, 0] =        float(event.metadata['year'][3])
        a[:, 1] = torch.tensor(event.nJet_selected)
        a[:, 2] = torch.tensor(event.xW)
        a[:, 3] = torch.tensor(event.xbW)

        e = torch.tensor(event.event) % 3

        for classifier in ['SvB', 'SvB_MA']:
            if classifier == 'SvB':
                c_logits, q_logits = self.classifier_SvB(j, o, a, e)
            if classifier == 'SvB_MA':
                c_logits, q_logits = self.classifier_SvB_MA(j, o, a, e)

            c_score, q_score = F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy()

            # classes = [mj,tt,zz,zh,hh]
            SvB = ak.zip({'pmj': c_score[:, 0],
                          'ptt': c_score[:, 1],
                          'pzz': c_score[:, 2],
                          'pzh': c_score[:, 3],
                          'phh': c_score[:, 4],
                          'q_1234': q_score[:, 0],
                          'q_1324': q_score[:, 1],
                          'q_1423': q_score[:, 2],
                          })

            SvB['ps'] = SvB.pzz + SvB.pzh + SvB.phh
            SvB['passMinPs'] = (SvB.pzz > 0.01) | (SvB.pzh > 0.01) | (SvB.phh > 0.01)
            SvB['zz'] = (SvB.pzz > SvB.pzh) & (SvB.pzz > SvB.phh)
            SvB['zh'] = (SvB.pzh > SvB.pzz) & (SvB.pzh > SvB.phh)
            SvB['hh'] = (SvB.phh > SvB.pzz) & (SvB.phh > SvB.pzh)

            error = ~np.isclose(event[classifier].ps, SvB.ps, atol=1e-5, rtol=1e-3)
            if np.any(error):
                delta = np.abs(event[classifier].ps - SvB.ps)
                worst = np.max(delta) == delta
                worst_event = event[worst][0]
                logging.warning(f'WARNING: Calculated {classifier} does not agree '
                                f'within tolerance for some events ({np.sum(error)}/{len(error)})', delta[worst])
                logging.warning('----------')
                for field in event[classifier].fields:
                    logging.warning(field, worst_event[classifier][field])
                logging.warning('----------')
                for field in SvB.fields:
                    logging.warning(f'{field}, {SvB[worst][field]}')

            # del event[classifier]
            event[classifier] = SvB

    def postprocess(self, accumulator):
        ...
