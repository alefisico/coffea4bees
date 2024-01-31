import pickle
import os
import time
import gc
import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
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

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector
from coffea import processor, util
from coffea.lumi_tools import LumiMask

from base_class.hist import Collection, Fill
from base_class.physics.object import LorentzVector, Jet, Muon, Elec

from analysis.helpers.MultiClassifierSchema import MultiClassifierSchema
from analysis.helpers.correctionFunctions import btagVariations
from analysis.helpers.correctionFunctions import btagSF_norm as btagSF_norm_file
from functools import partial
from multiprocessing import Pool

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
# print(torch.__config__.parallel_info())

from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.common import init_jet_factory, jet_corrections, mask_event_decision, apply_btag_sf, drClean
import logging


#
# Setup
#
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")
ak.behavior.update(vector.behavior)

from base_class.hist import H, Template

class SvBHists(Template):
    ps      = H((100, 0, 1, ('ps', "Regressed P(Signal)")))
    ptt     = H((100, 0, 1, ('ptt', "Regressed P(tT)")))

    ps_zz   = H((100, 0, 1, ('ps_zz', "Regressed P(Signal) $|$ P(ZZ) is largest ")))
    ps_zh   = H((100, 0, 1, ('ps_zh', "Regressed P(Signal) $|$ P(ZH) is largest ")))
    ps_hh   = H((100, 0, 1, ('ps_hh', "Regressed P(Signal) $|$ P(HH) is largest ")))


class FvTHists(Template):
    FvT  = H((100, 0, 5, ('FvT', 'FvT reweight')))
    pd4  = H((100, 0, 1, ("pd4",   'FvT Regressed P(Four-tag Data)')))
    pd3  = H((100, 0, 1, ("pd3",   'FvT Regressed P(Three-tag Data)')))
    pt4  = H((100, 0, 1, ("pt4",   'FvT Regressed P(Four-tag t#bar{t})')))
    pt3  = H((100, 0, 1, ("pt3",   'FvT Regressed P(Three-tag t#bar{t})')))
    pm4  = H((100, 0, 1, ("pm4",   'FvT Regressed P(Four-tag Multijet)')))
    pm3  = H((100, 0, 1, ("pm3",   'FvT Regressed P(Three-tag Multijet)')))
    pt   = H((100, 0, 1, ("pt",    'FvT Regressed P(t#bar{t})')))
    std  = H((100, 0, 3, ("std",   'FvT Standard Deviation')))
    frac_err = H((100, 0, 5, ("frac_err",  'FvT std/FvT')))
    #'q_1234', 'q_1324', 'q_1423',

class QuadJetHists(Template):
    dr              = H((50,     0, 5,   ("dr",          'Diboson Candidate $\\Delta$R(d,d)')))
    dphi            = H((100, -3.2, 3.2, ("dphi",        'Diboson Candidate $\\Delta$R(d,d)')))
    deta            = H((100,   -5, 5,   ("deta",        'Diboson Candidate $\\Delta$R(d,d)')))
    FvT_score       = H((100, 0, 1,      ("FvT_q_score", 'Diboson FvT q score')))
    SvB_q_score     = H((100, 0, 1,      ("SvB_q_score", 'Diboson SvB q score')))
    SvB_MA_q_score  = H((100, 0, 1,      ("SvB_q_score", 'Diboson SvB MA q score')))
    xZZ             = H((100, 0, 10,     ("xZZ",         'Diboson Candidate zZZ')))
    xZH             = H((100, 0, 10,     ("xZH",         'Diboson Candidate zZH')))
    xHH             = H((100, 0, 10,     ("xHH",         'Diboson Candidate zHH')))

    lead_vs_subl_m   = H((50, 0, 250, ('lead.mass', 'Lead Boson Candidate Mass')),
                         (50, 0, 250, ('subl.mass', 'Subl Boson Candidate Mass')))

    close_vs_other_m = H((50, 0, 250, ('close.mass', 'Close Boson Candidate Mass')),
                         (50, 0, 250, ('other.mass', 'Other Boson Candidate Mass')))

    lead            = LorentzVector.plot_pair(('...', R'Lead Boson Candidate'),  'lead',  skip=['n'])
    subl            = LorentzVector.plot_pair(('...', R'Subl Boson Candidate'),  'subl',  skip=['n'])
    close           = LorentzVector.plot_pair(('...', R'Close Boson Candidate'), 'close', skip=['n'])
    other           = LorentzVector.plot_pair(('...', R'Other Boson Candidate'), 'other', skip=['n'])




class cutFlow:

    def __init__(self, cuts):
        self._cutFlowThreeTag = {}
        self._cutFlowFourTag  = {}

        for c in cuts:
            self._cutFlowThreeTag[c] = (0, 0)    # weighted, raw
            self._cutFlowFourTag [c] = (0, 0)    # weighted, raw

    def fill(self, cut, event, allTag=False, wOverride=None):

        if allTag:

            if wOverride:
                sumw = wOverride
            else:
                sumw = np.sum(event.weight)

            sumn_3, sumn_4 = len(event), len(event)
            sumw_3, sumw_4 = sumw, sumw
        else:
            e3, e4 = event[event.threeTag], event[event.fourTag]

            sumw_3 = np.sum(e3.weight)
            sumn_3 = len(e3.weight)

            sumw_4 = np.sum(e4.weight)
            sumn_4 = len(e4.weight)

        self._cutFlowThreeTag[cut] = (sumw_3, sumn_3)     # weighted, raw
        self._cutFlowFourTag [cut] = (sumw_4, sumn_4)     # weighted, raw


    def addOutput(self, o, dataset):

        o["cutFlowFourTag"] = {}
        o["cutFlowFourTagUnitWeight"] = {}
        o["cutFlowFourTag"][dataset] = {}
        o["cutFlowFourTagUnitWeight"][dataset] = {}
        for k, v in  self._cutFlowFourTag.items():
            o["cutFlowFourTag"][dataset][k] = v[0]
            o["cutFlowFourTagUnitWeight"][dataset][k] = v[1]

        o["cutFlowThreeTag"] = {}
        o["cutFlowThreeTagUnitWeight"] = {}
        o["cutFlowThreeTag"][dataset] = {}
        o["cutFlowThreeTagUnitWeight"][dataset] = {}
        for k, v in  self._cutFlowThreeTag.items():
            o["cutFlowThreeTag"][dataset][k] = v[0]
            o["cutFlowThreeTagUnitWeight"][dataset][k] = v[1]

        return


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
    def __init__(self, *, JCM = '', addbtagVariations=None, SvB=None, SvB_MA=None, threeTag = True, apply_puWeight = False, apply_prefire = False, apply_trigWeight = True, apply_btagSF = True, apply_FvT = True, regions=['SR'], corrections_metadata='analysis/metadata/corrections.yml',  btagSF=True):
        logging.debug('\nInitialize Analysis Processor')
        self.blind = False
        print('Initialize Analysis Processor')
        self.cutFlowCuts = ["all", "passHLT", "passMETFilter", "passJetMult", "passJetMult_btagSF", "passPreSel", "passDiJetMass", 'SR', 'SB', 'passSvB', 'failSvB']
        self.histCuts = ['passPreSel', 'passSvB', 'failSvB']
        self.doThreeTag = threeTag
        self.tags = ['threeTag', 'fourTag'] if threeTag else ['fourTag']
        self.regions = regions
        self.signals = ['zz', 'zh', 'hh']
        self.JCM = jetCombinatoricModel(JCM)
        self.apply_FvT = apply_FvT
        self.btagVar = btagVariations(systematics=addbtagVariations)  #### AGE: these two need to be review later
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        self.apply_puWeight = apply_puWeight
        self.apply_prefire  = apply_prefire
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, 'r'))
        self.btagSF  = btagSF

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
        btagSF_norm = btagSF_norm_file(dataset)
        nEvent = len(event)

        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = nEvent

        #
        #  Cut Flows
        #
        self._cutFlow = cutFlow(self.cutFlowCuts)

        juncWS = [ self.corrections_metadata[year]["JERC"][0].replace('STEP', istep)
                   for istep in ['L1FastJet', 'L2Relative', 'L2L3Residual', 'L3Absolute'] ]      ###### AGE: to be reviewed for data, but should be remove with jsonpog
        if isMC:
            juncWS += self.corrections_metadata[year]["JERC"][1:]

        #
        #  Turn blinding off for mixing
        #
        if dataset.find("mixed") != -1:
            self.blind = False

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


        #    xWt0 = dir.make<TH1F>("xWt0", (name+"/xWt0; X_{Wt,0}; Entries").c_str(), 60, 0, 12);
        #    xWt1 = dir.make<TH1F>("xWt1", (name+"/xWt1; X_{Wt,1}; Entries").c_str(), 60, 0, 12);
        #    //xWt2 = dir.make<TH1F>("xWt2", (name+"/xWt2; X_{Wt,2}; Entries").c_str(), 60, 0, 12);
        #    xWt  = dir.make<TH1F>("xWt",  (name+"/xWt;  X_{Wt};   Entries").c_str(), 60, 0, 12);
        #    t0 = new trijetHists(name+"/t0",  fs, "Top Candidate (#geq0 non-candidate jets)");
        #    t1 = new trijetHists(name+"/t1",  fs, "Top Candidate (#geq1 non-candidate jets)");
        #    //t2 = new trijetHists(name+"/t2",  fs, "Top Candidate (#geq2 non-candidate jets)");
        #    t = new trijetHists(name+"/t",  fs, "Top Candidate");


        fill += hist.add('nPVs',     (101, -0.5, 100.5, ('PV.npvs',     'Number of Primary Vertices')))
        fill += hist.add('nPVsGood', (101, -0.5, 100.5, ('PV.npvsGood', 'Number of Good Primary Vertices')))

        fill += hist.add('hT',          (100,  0,   1000,  ('hT',          'H_{T} [GeV}')))
        fill += hist.add('hT_selected', (100,  0,   1000,  ('hT_selected', 'H_{T} (selected jets) [GeV}')))

        #
        #  Make quad jet hists
        #
        fill += LorentzVector.plot_pair(('v4j', R'$HH_{4b}$'), 'v4j', skip=['n', 'dr', 'dphi', 'st'], bins={'mass': (120, 0, 1200)})
        fill += QuadJetHists(('quadJet_selected', 'Selected Quad Jet'), 'quadJet_selected')
        fill += QuadJetHists(('quadJet_min_dr',   'Min dR Quad Jet'),   'quadJet_min_dr')

        #
        #  Make classifier hists
        #
        fill += FvTHists(('FvT', 'FvT Classifier'), 'FvT')
        fill += hist.add('FvT_noFvT', (100, 0, 5, ('FvT.FvT', 'FvT reweight')), weight="weight_noFvT")

        fill += SvBHists(('SvB', 'SvB Classifier'), 'SvB')
        fill += SvBHists(('SvB_MA', 'SvB MA Classifier'), 'SvB_MA')

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
        #  Config weights
        #
        self.apply_puWeight   = (self.apply_puWeight  ) and isMC
        self.apply_prefire    = (self.apply_prefire   ) and isMC and ('L1PreFiringWeight' in event.fields) and (year != 'UL18')
        self.apply_trigWeight = (self.apply_trigWeight) and isMC and ('trigWeight' in event.fields)
        self.apply_btagSF     = (self.apply_btagSF)     and isMC and (self.btagSF is not None)

        logging.debug(fname)
        logging.debug(f'{chunk}Process {nEvent} Events')

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split('/')[-1], '')
        event['FvT']    = NanoEventsFactory.from_root(f'{path}{"FvT_3bDvTMix4bDvT_v0_newSB.root" if "mix" in dataset else "FvT.root"}',
                                                      entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().FvT
        event['FvT', 'frac_err'] = event['FvT'].std / event['FvT'].FvT

        event['SvB']    = NanoEventsFactory.from_root(f'{path}{"SvB_newSBDef.root" if "mix" in dataset else "SvB.root"}',
                                                      entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB

        event['SvB_MA'] = NanoEventsFactory.from_root(f'{path}{"SvB_MA_newSBDef.root" if "mix" in dataset else "SvB_MA.root"}',
                                                      entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB_MA


        if not ak.all(event.SvB.event == event.event):
            logging.error('ERROR: SvB events do not match events ttree')
            return

        if not ak.all(event.SvB_MA.event == event.event):
            logging.error('ERROR: SvB_MA events do not match events ttree')
            return

        if not ak.all(event.FvT.event == event.event):
            logging.error('ERROR: FvT events do not match events ttree')
            return

        #
        # defining SvB for different SR
        #
        setSvBVars("SvB",    event)
        setSvBVars("SvB_MA", event)

        if isMC:
            self._cutFlow.fill("all",  event, allTag=True, wOverride=(lumi * xs * kFactor))
        else:
            lumimask = LumiMask(self.corrections_metadata[year]['goldenJSON'])
            jsonFilter = np.array( lumimask(event.run, event.luminosityBlock) )
            event = event[jsonFilter]
            self._cutFlow.fill("all",  event, allTag=True)

        #
        # Get trigger decisions
        #
        event['passHLT'] = mask_event_decision( event, decision="OR", branch="HLT", list_to_mask=event.metadata['trigger']  )

        if not isMC and 'mix' not in dataset:      # for data, apply trigger cut first thing, for MC, keep all events and apply trigger in cutflow and for plotting
            event = event[event.passHLT]

        #
        # weights
        #
        if isMC:
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)
            logging.debug(f"event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw) = {event.genWeight[0]} * ({lumi} * {xs} * {kFactor} / {genEventSumw}) = {event.weight[0]}\n")
            if self.apply_trigWeight:
                event['weight'] = event.weight * event.trigWeight.Data
        else:
            event['weight'] = 1
            # logging.info(f"event['weight'] = {event.weight}")

        self._cutFlow.fill("passHLT",  event, allTag=True)

        #
        # METFilter
        #
        passMETFilter = mask_event_decision( event, decision="AND", branch="Flag",
                                            list_to_mask=self.corrections_metadata[year]['METFilter'],
                                            list_to_skip=['BadPFMuonDzFilter', 'hfNoisyHitsFilter']  )
        event['passMETFilter'] = passMETFilter

        event = event[event.passMETFilter] # HACK
        self._cutFlow.fill("passMETFilter",  event, allTag=True)

        #
        # Lepton Selction
        #

        #
        # Adding muons (loose muon id definition)
        # https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        #
        #event['Muon', 'selected'] = (event.Muon.pt > 10) & (abs(event.Muon.eta) < 2.5) & (event.Muon.pfRelIso04_all < 0.15) & (event.Muon.looseId)
        event['Muon', 'selected'] = (event.Muon.pt > 25) & (abs(event.Muon.eta) < 2.4) & (event.Muon.pfRelIso04_all < 0.25) & (event.Muon.looseId)
        event['nMuon_selected'] = ak.sum(event.Muon.selected, axis=1)
        event['selMuon'] = event.Muon[event.Muon.selected]

        #
        # Adding electrons (loose electron id)
        # https://twiki.cern.ch/twiki/bin/view/CMS/CutBasedElectronIdentificationRun2
        #
        #event['Electron', 'selected'] = (event.Electron.pt > 15) & (abs(event.Electron.eta) < 2.5) & (event.Electron.pfRelIso03_all < 0.15) & (event.Electron.mvaIso_WP90)
        event['Electron', 'selected'] = (event.Electron.pt > 25) & (abs(event.Electron.eta) < 2.5) & (event.Electron.cutBased >= 2)
        event['nElectron_selected'] = ak.sum(event.Electron.selected, axis=1)
        event['selElec'] = event.Electron[event.Electron.selected]

        #
        # Calculate and apply Jet Energy Calibration   ## AGE: currently not applying to data and mixeddata
        #
        if isMC and juncWS is not None:
            jet_variations = init_jet_factory(juncWS, event)  #### currently creates the pt_raw branch
#            jet_tmp = jet_corrections( event.Jet, event.fixedGridRhoFastjetAll, jec_type=['L1L2L3Res'])   # AGE: jsonpog+correctionlib but not final, that is why it is not used yet

        event['Jet', 'calibration'] = event.Jet.pt / ( 1 if 'data' in dataset else event.Jet.pt_raw )    # AGE: I include the mix condition, I think it is wrong, to check later
        # print(f'calibration nominal: \n{ak.mean(event.Jet.calibration)}')
        selLepton = ak.concatenate( [event.selElec, event.selMuon], axis=1 )
        event['Jet', 'lepton_cleaned'] = drClean( event.Jet, selLepton )[1]  ### 0 is the collection of jets, 1 is the flag

        event['Jet', 'pileup'] = ((event.Jet.puId < 0b110) & (event.Jet.pt < 50)) | ((np.abs(event.Jet.eta) > 2.4) & (event.Jet.pt < 40))
        event['Jet', 'selected_loose'] = (event.Jet.pt >= 20) & ~event.Jet.pileup & event.Jet.lepton_cleaned
        event['Jet', 'selected'] = (event.Jet.pt >= 40) & (np.abs(event.Jet.eta) <= 2.4) & ~event.Jet.pileup & event.Jet.lepton_cleaned
        event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)
        event['selJet'] = event.Jet[event.Jet.selected]


        selev = event[event.nJet_selected >= 4]
        self._cutFlow.fill("passJetMult",  selev, allTag=True)
        
        for iEvent in range(10):
            logging.info(f'{chunk} event idx ={iEvent} selectedJets pt {selev[iEvent].Jet[selev[iEvent].Jet.selected].pt}\n')
            logging.info(f'{chunk} event idx ={iEvent} selectedJets eta {selev[iEvent].Jet[selev[iEvent].Jet.selected].eta}\n')
            logging.info(f'{chunk} event idx ={iEvent} selectedJets phi {selev[iEvent].Jet[selev[iEvent].Jet.selected].phi}\n')
            logging.info(f'{chunk} event idx ={iEvent} selectedJets mass {selev[iEvent].Jet[selev[iEvent].Jet.selected].mass}\n')
            logging.info(f'{chunk} event idx ={iEvent} selectedJets btagDeepFlavB {selev[iEvent].Jet[selev[iEvent].Jet.selected].btagDeepFlavB}\n')
            logging.info(f'{chunk} event idx ={iEvent} selectedJets bRegCorr {selev[iEvent].Jet[selev[iEvent].Jet.selected].bRegCorr}\n')
            logging.info(f'{chunk} event idx ={iEvent} xbW {selev[iEvent].xbW}\n')
            logging.info(f'{chunk} event idx ={iEvent} xW {selev[iEvent].xW}\n')


        logging.info(f'{chunk} {type(event.Jet)}\n')        

        selev['Jet', 'tagged']       = selev.Jet.selected & (selev.Jet.btagDeepFlavB >= 0.6)
        selev['Jet', 'tagged_loose'] = selev.Jet.selected & (selev.Jet.btagDeepFlavB >= 0.3)
        selev['nJet_tagged']         = ak.num(selev.Jet[selev.Jet.tagged])
        selev['nJet_tagged_loose']   = ak.num(selev.Jet[selev.Jet.tagged_loose])
        selev['tagJet']              = selev.Jet[selev.Jet.tagged]

        fourTag  = (selev['nJet_tagged']       >= 4)
        threeTag = (selev['nJet_tagged_loose'] == 3) & (selev['nJet_selected'] >= 4)


        selev[ 'fourTag']   =  fourTag
        selev['threeTag']   = threeTag

        # selev['tag'] = ak.Array({'threeTag':selev.threeTag, 'fourTag':selev.fourTag})
        selev['passPreSel'] = selev.threeTag | selev.fourTag

        tagCode = np.full(len(selev), 0, dtype=int)
        tagCode[selev.fourTag]  = 4
        tagCode[selev.threeTag] = 3
        selev['tag'] = tagCode

        #
        # Calculate and apply pileup weight, L1 prefiring weight
        #
        if self.apply_puWeight:
            puWeight = list(correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['PU']).values())[0]
            for var in ['nominal', 'up', 'down']:
                selev[f'PU_weight_{var}'] = puWeight.evaluate(selev.Pileup.nTrueInt.to_numpy(), var)
            selev['weight'] = selev.weight * selev.PU_weight_nominal

        if self.apply_prefire:
            selev['weight'] = selev.weight * selev.L1PreFiringWeight.Nom

        #
        # Calculate and apply btag scale factors
        #
        if self.apply_btagSF:
            btagSF = correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['btagSF'])['deepJet_shape']
            selev['weight'] = apply_btag_sf(selev, selev.selJet,
                                            correction_file=self.corrections_metadata[year]['btagSF'],
                                            btag_var=self.btagVar,
                                            btagSF_norm=btagSF_norm,
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

        # print(f'{ak.mean(canJet.calibration)} (canJets)')
        # print(canJet_idx[0])
        # print(selev[0].Jet[canJet_idx[0]].pt)
        # print(selev[0].Jet[canJet_idx[0]].bRegCorr)
        # print(selev[0].Jet[canJet_idx[0]].calibration)

        if self.doThreeTag:

            #
            # calculate pseudoTagWeight for threeTag events
            #
            selev['Jet_untagged_loose'] = selev.Jet[selev.Jet.selected & ~selev.Jet.tagged_loose]
            nJet_pseudotagged = np.zeros(len(selev), dtype=int)
            pseudoTagWeight = np.ones(len(selev))
            pseudoTagWeight[selev.threeTag], nJet_pseudotagged[selev.threeTag] = self.JCM(selev[selev.threeTag]['Jet_untagged_loose'])
            selev['nJet_pseudotagged'] = nJet_pseudotagged

            # check that pseudoTagWeight calculation agrees with c++
            selev.issue = (abs(selev.pseudoTagWeight - pseudoTagWeight) / selev.pseudoTagWeight > 0.0001) & (pseudoTagWeight != 1)

            # add pseudoTagWeight to event
            selev['pseudoTagWeight'] = pseudoTagWeight

            #
            # apply pseudoTagWeight and FvT to threeTag events
            #
            weight_noJCM_noFvT = selev.weight
            selev['weight_noJCM_noFvT'] = weight_noJCM_noFvT

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
        # CutFlow
        #
        self._cutFlow.fill("passPreSel", selev)

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

        quadJet['FvT_q_score'] = np.concatenate((np.reshape(np.array(selev.FvT.q_1234), (-1, 1)),
                                                 np.reshape(np.array(selev.FvT.q_1324), (-1, 1)),
                                                 np.reshape(np.array(selev.FvT.q_1423), (-1, 1))), axis=1)

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
        selev['passSvB'] = (selev['SvB_MA'].ps > 0.80)
        selev['failSvB'] = (selev['SvB_MA'].ps < 0.05)

        #
        # Blind data in fourTag SR
        #
        if not (isMC or 'mixed' in dataset) and self.blind:
            selev = selev[~(selev['quadJet_selected'].SR & selev.fourTag)]

        if self.classifier_SvB is not None:
            self.compute_SvB(selev)

        #
        # fill histograms
        #
        self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
        self._cutFlow.fill("SR",            selev[(selev.passDiJetMass & selev['quadJet_selected'].SR)])
        self._cutFlow.fill("SB",            selev[(selev.passDiJetMass & selev['quadJet_selected'].SB)])
        self._cutFlow.fill("passSvB",       selev[selev.passSvB])
        self._cutFlow.fill("failSvB",       selev[selev.failSvB])

        # fill.cache(selev)
        fill(selev)

        garbage = gc.collect()
        # print('Garbage:',garbage)

        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f'{chunk}{nEvent/elapsed:,.0f} events/s')

        self._cutFlow.addOutput(processOutput, event.metadata['dataset'])

        return hist.output | processOutput


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
