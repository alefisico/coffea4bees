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
import cachetools
import yaml
import warnings
import torch
import torch.nn.functional as F

from analysis.helpers.networks import HCREnsemble

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from coffea.nanoevents.methods import vector
from coffea import processor, util

from base_class.hist import Collection, Fill
from base_class.aktools import where
from base_class.physics.object import LorentzVector, Jet

from analysis.helpers.MultiClassifierSchema import MultiClassifierSchema
from analysis.helpers.correctionFunctions import btagVariations, juncVariations
from analysis.helpers.correctionFunctions import btagSF_norm as btagSF_norm_file
from functools import partial
from multiprocessing import Pool

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
# print(torch.__config__.parallel_info())

from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.common import init_jet_factory, jet_corrections
import logging


#
# Setup
#
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")
ak.behavior.update(vector.behavior)


@dataclass
class variable:
    def __init__(self, name, bins, label='Events'):
        self.name = name
        self.bins = bins
        self.label = label

class cutFlow:

    def __init__(self, cuts):
        self._cutFlowThreeTag = {}
        self._cutFlowFourTag  = {}

        for c in cuts:
            self._cutFlowThreeTag[c] = (0, 0) # weighted, raw
            self._cutFlowFourTag [c] = (0, 0) # weighted, raw

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

        self._cutFlowThreeTag[cut] = (sumw_3, sumn_3) # weighted, raw
        self._cutFlowFourTag [cut] = (sumw_4, sumn_4) # weighted, raw


    def addOutput(self, o, dataset):

        o["cutFlowFourTag"] = {}
        o["cutFlowFourTagUnitWeight"] = {}
        o["cutFlowFourTag"][dataset] = {}
        o["cutFlowFourTagUnitWeight"][dataset] = {}
        for k,v in  self._cutFlowFourTag.items():
            o["cutFlowFourTag"][dataset][k] = v[0]
            o["cutFlowFourTagUnitWeight"][dataset][k] = v[1]

        o["cutFlowThreeTag"] = {}
        o["cutFlowThreeTagUnitWeight"] = {}
        o["cutFlowThreeTag"][dataset] = {}
        o["cutFlowThreeTagUnitWeight"][dataset] = {}
        for k,v in  self._cutFlowThreeTag.items():
            o["cutFlowThreeTag"][dataset][k] = v[0]
            o["cutFlowThreeTagUnitWeight"][dataset][k] = v[1]

        return


def setSvBVars(SvBName, event):
    largest_name = np.array(['None', 'ZZ', 'ZH', 'HH'])

    event[SvBName, 'passMinPs'] = (getattr(event, SvBName).pzz>0.01) | (getattr(event, SvBName).pzh>0.01) | (getattr(event, SvBName).phh>0.01)
    event[SvBName, 'zz'] = (getattr(event, SvBName).pzz >  getattr(event, SvBName).pzh) & (getattr(event, SvBName).pzz >  getattr(event, SvBName).phh)
    event[SvBName, 'zh'] = (getattr(event, SvBName).pzh >  getattr(event, SvBName).pzz) & (getattr(event, SvBName).pzh >  getattr(event, SvBName).phh)
    event[SvBName, 'hh'] = (getattr(event, SvBName).phh >= getattr(event, SvBName).pzz) & (getattr(event, SvBName).phh >= getattr(event, SvBName).pzh)
    event[SvBName, 'largest'] = largest_name[ getattr(event, SvBName).passMinPs*(1*getattr(event, SvBName).zz + 2*getattr(event, SvBName).zh + 3*getattr(event, SvBName).hh) ]

    #
    #  Set ps_{bb}
    #
    event[SvBName, 'ps_zz'] = where(~getattr(event, SvBName).passMinPs, (~getattr(event, SvBName).passMinPs, -4))
    event[SvBName, 'ps_zh'] = where(~getattr(event, SvBName).passMinPs, (~getattr(event, SvBName).passMinPs, -4))
    event[SvBName, 'ps_hh'] = where(~getattr(event, SvBName).passMinPs, (~getattr(event, SvBName).passMinPs, -4))

    event[SvBName, 'ps_zz'] = where((getattr(event, SvBName).passMinPs),
                                    (getattr(event, SvBName).zz, getattr(event, SvBName).pzz),
                                    (getattr(event, SvBName).zh, -2),
                                    (getattr(event, SvBName).hh, -3))

    event[SvBName, 'ps_zh'] = where((getattr(event, SvBName).passMinPs),
                                    (getattr(event, SvBName).zz, -1),
                                    (getattr(event, SvBName).zh, getattr(event, SvBName).pzh),
                                    (getattr(event, SvBName).hh, -3))

    event[SvBName, 'ps_hh'] = where((getattr(event, SvBName).passMinPs),
                                    (getattr(event, SvBName).zz, -1),
                                    (getattr(event, SvBName).zh, -2),
                                    (getattr(event, SvBName).hh, getattr(event, SvBName).phh))



# def count_nested_dict(nested_dict, c=0):
#     for key in nested_dict:
#         if isinstance(nested_dict[key], dict):
#             c = count_nested_dict(nested_dict[key], c)
#         else:
#             c += 1
#     return c

class analysis(processor.ProcessorABC):
    def __init__(self, *, JCM = '', addbtagVariations=None, addjuncVariations=None, SvB=None, SvB_MA=None, threeTag = True, apply_puWeight = False, apply_prefire = False, apply_trigWeight = True, apply_btagSF = True, regions=['SR'], corrections_metadata='analysis/metadata/corrections.yml',  btagSF=True):
        logging.debug('\nInitialize Analysis Processor')
        self.blind = False
        print('Initialize Analysis Processor')
        self.cutFlowCuts = ["all","passHLT","passMETFilter","passJetMult","passJetMult_btagSF","passPreSel","passDiJetMass",'SR','SB','passSvB','failSvB']
        self.histCuts = ['passPreSel','passSvB','failSvB']
        self.threeTag = threeTag
        self.tags = ['threeTag','fourTag'] if threeTag else ['fourTag']
        self.regions = regions
        self.signals = ['zz','zh','hh']
        self.JCM = jetCombinatoricModel(JCM)
        self.doReweight = True
        self.btagVar = btagVariations(systematics=addbtagVariations)  #### AGE: these two need to be review later
        self.juncVar = juncVariations(systematics=addjuncVariations)
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        self.apply_puWeight = apply_puWeight
        self.apply_prefire  = apply_prefire
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, 'r'))
        self.btagSF  = btagSF


        self.variables = []
        self.variables_systematics = self.variables[0:8]
        #jet_extras = [variable('calibration', hist.Bin('x','Calibration Factor', 20, 0, 2))]
        #self.variables += fourvectorhists('canJet', 'Boson Candidate Jets', mass=(50, 0, 50), label='Jets', extras=jet_extras)




    def process(self, event):
        tstart = time.time()

        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        era     = event.metadata.get('era','')
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        btagSF_norm = btagSF_norm_file(dataset)
        nEvent = len(event)
        np.random.seed(0)

        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = nEvent

        #
        #  Cut Flows
        #
        self._cutFlow            = cutFlow(self.cutFlowCuts)

        puWeight= self.corrections_metadata[year]['PU']
        juncWS = [ self.corrections_metadata[year]["JERC"][0].replace('STEP', istep) for istep in ['L1FastJet', 'L2Relative', 'L2L3Residual', 'L3Absolute'] ]  ###### AGE: to be reviewed for data, but should be remove with jsonpog
        if isMC: juncWS += self.corrections_metadata[year]["JERC"][1:]

        #
        #  Turn blinding off for mixing
        #
        if dataset.find("mixed") != -1:
            self.blind = False

        #
        # Hists
        #
        fill = Fill(process = processName, year = year, weight = 'weight')

        hist = Collection(process = [processName],
                          year    = [year],
                          tag     = [3,4,0], # 3 / 4/ Other
                          region  = [2,1,0], # SR / SB / Other
                          **dict((s, ...) for s in self.histCuts))



        #
        # To Add
        #
        #    nSelJetsUnweighted = dir.make<TH1F>("nSelJetsUnweighted", (name+"/nSelJetsUnweighted; Number of Selected Jets (Unweighted); Entries").c_str(),  16,-0.5,15.5);
        #    nTagJets = dir.make<TH1F>("nTagJets", (name+"/nTagJets; Number of Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
        #    nTagJetsUnweighted = dir.make<TH1F>("nTagJetsUnweighted", (name+"/nTagJets; Number of Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
        #    nPSTJets = dir.make<TH1F>("nPSTJets", (name+"/nPSTJets; Number of Tagged + Pseudo-Tagged Jets; Entries").c_str(),  16,-0.5,15.5);
        #    tagJets = new jetHists(name+"/tagJets", fs, "Tagged Jets");
        
        #    nAllMuons = dir.make<TH1F>("nAllMuons", (name+"/nAllMuons; Number of Muons (no selection); Entries").c_str(),  6,-0.5,5.5);
        #    nIsoMed25Muons = dir.make<TH1F>("nIsoMed25Muons", (name+"/nIsoMed25Muons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
        #    nIsoMed40Muons = dir.make<TH1F>("nIsoMed40Muons", (name+"/nIsoMed40Muons; Number of Prompt Muons; Entries").c_str(),  6,-0.5,5.5);
        #    allMuons        = new muonHists(name+"/allMuons", fs, "All Muons");
        #    muons_isoMed25  = new muonHists(name+"/muon_isoMed25", fs, "iso Medium 25 Muons");
        #    muons_isoMed40  = new muonHists(name+"/muon_isoMed40", fs, "iso Medium 40 Muons");
        
        
        #    nAllElecs = dir.make<TH1F>("nAllElecs", (name+"/nAllElecs; Number of Elecs (no selection); Entries").c_str(),  16,-0.5,15.5);
        #    nIsoMed25Elecs = dir.make<TH1F>("nIsoMed25Elecs", (name+"/nIsoMed25Elecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
        #    nIsoMed40Elecs = dir.make<TH1F>("nIsoMed40Elecs", (name+"/nIsoMed40Elecs; Number of Prompt Elecs; Entries").c_str(),  6,-0.5,5.5);
        #    allElecs        = new elecHists(name+"/allElecs", fs, "All Elecs");
        #    elecs_isoMed25  = new elecHists(name+"/elec_isoMed25", fs, "iso Medium 25 Elecs");
        #    elecs_isoMed40  = new elecHists(name+"/elec_isoMed40", fs, "iso Medium 40 Elecs");
        #  
        
        #    leadSt_m_vs_sublSt_m = dir.make<TH2F>("leadSt_m_vs_sublSt_m", (name+"/leadSt_m_vs_sublSt_m; S_{T} leading boson candidate Mass [GeV]; S_{T} subleading boson candidate Mass [GeV]; Entries").c_str(), 50,0,250, 50,0,250);
        #    m4j_vs_leadSt_dR = dir.make<TH2F>("m4j_vs_leadSt_dR", (name+"/m4j_vs_leadSt_dR; m_{4j} [GeV]; S_{T} leading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);
        #    m4j_vs_sublSt_dR = dir.make<TH2F>("m4j_vs_sublSt_dR", (name+"/m4j_vs_sublSt_dR; m_{4j} [GeV]; S_{T} subleading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);

        #    close  = new dijetHists(name+"/close",  fs,               "Minimum #DeltaR(j,j) Dijet");
        #    other  = new dijetHists(name+"/other",  fs, "Complement of Minimum #DeltaR(j,j) Dijet");
        #    close_m_vs_other_m = dir.make<TH2F>("close_m_vs_other_m", (name+"/close_m_vs_other_m; Minimum #DeltaR(j,j) Dijet Mass [GeV]; Complement of Minimum #DeltaR(j,j) Dijet Mass [GeV]; Entries").c_str(), 50,0,250, 50,0,250);

        #    st = dir.make<TH1F>("st", (name+"/st; Scalar sum of jet p_{T}'s [GeV]; Entries").c_str(), 130, 200, 1500);


#    xWt0 = dir.make<TH1F>("xWt0", (name+"/xWt0; X_{Wt,0}; Entries").c_str(), 60, 0, 12);
#    xWt1 = dir.make<TH1F>("xWt1", (name+"/xWt1; X_{Wt,1}; Entries").c_str(), 60, 0, 12);
#    //xWt2 = dir.make<TH1F>("xWt2", (name+"/xWt2; X_{Wt,2}; Entries").c_str(), 60, 0, 12);
#    xWt  = dir.make<TH1F>("xWt",  (name+"/xWt;  X_{Wt};   Entries").c_str(), 60, 0, 12);
#    t0 = new trijetHists(name+"/t0",  fs, "Top Candidate (#geq0 non-candidate jets)");
#    t1 = new trijetHists(name+"/t1",  fs, "Top Candidate (#geq1 non-candidate jets)");
#    //t2 = new trijetHists(name+"/t2",  fs, "Top Candidate (#geq2 non-candidate jets)");
#    t = new trijetHists(name+"/t",  fs, "Top Candidate");
#  


#    FvT = dir.make<TH1F>("FvT", (name+"/FvT; Kinematic Reweight; Entries").c_str(), 100, 0, 5);
#    FvTUnweighted = dir.make<TH1F>("FvTUnweighted", (name+"/FvTUnweighted; Kinematic Reweight; Entries").c_str(), 100, 0, 5);
#    FvT_pd4 = dir.make<TH1F>("FvT_pd4", (name+"/FvT_pd4; FvT Regressed P(Four-tag Data) ; Entries").c_str(), 100, 0, 1);
#    FvT_pd3 = dir.make<TH1F>("FvT_pd3", (name+"/FvT_pd3; FvT Regressed P(Three-tag Data) ; Entries").c_str(), 100, 0, 1);
#    FvT_pt4 = dir.make<TH1F>("FvT_pt4", (name+"/FvT_pt4; FvT Regressed P(Four-tag t#bar{t}) ; Entries").c_str(), 100, 0, 1);
#    FvT_pt3 = dir.make<TH1F>("FvT_pt3", (name+"/FvT_pt3; FvT Regressed P(Three-tag t#bar{t}) ; Entries").c_str(), 100, 0, 1);
#    FvT_pm4 = dir.make<TH1F>("FvT_pm4", (name+"/FvT_pm4; FvT Regressed P(Four-tag Multijet) ; Entries").c_str(), 100, 0, 1);
#    FvT_pm3 = dir.make<TH1F>("FvT_pm3", (name+"/FvT_pm3; FvT Regressed P(Three-tag Multijet) ; Entries").c_str(), 100, 0, 1);
#    FvT_pt  = dir.make<TH1F>("FvT_pt",  (name+"/FvT_pt;  FvT Regressed P(t#bar{t}) ; Entries").c_str(), 100, 0, 1);
#    FvT_std = dir.make<TH1F>("FvT_std",  (name+"/FvT_pt;  FvT Standard Deviation ; Entries").c_str(), 100, 0, 5);
#    FvT_ferr = dir.make<TH1F>("FvT_ferr",  (name+"/FvT_ferr;  FvT std/FvT ; Entries").c_str(), 100, 0, 5);


#  SvB and SvB MA
#    SvB_ps  = dir.make<TH1F>("SvB_ps",  (name+"/SvB_ps;  SvB Regressed P(ZZ)+P(ZH); Entries").c_str(), 100, 0, 1);
#    SvB_pzz = dir.make<TH1F>("SvB_pzz", (name+"/SvB_pzz; SvB Regressed P(ZZ); Entries").c_str(), 100, 0, 1);
#    SvB_pzh = dir.make<TH1F>("SvB_pzh", (name+"/SvB_pzh; SvB Regressed P(ZH); Entries").c_str(), 100, 0, 1);
#    SvB_phh = dir.make<TH1F>("SvB_phh", (name+"/SvB_phh; SvB Regressed P(HH); Entries").c_str(), 100, 0, 1);
#    SvB_ptt = dir.make<TH1F>("SvB_ptt", (name+"/SvB_ptt; SvB Regressed P(t#bar{t}); Entries").c_str(), 100, 0, 1);
#    SvB_ps_hh = dir.make<TH1F>("SvB_ps_hh",  (name+"/SvB_ps_hh;  SvB Regressed P(Signal), P(HH) is largest; Entries").c_str(), 100, 0, 1);
#    SvB_ps_zh = dir.make<TH1F>("SvB_ps_zh",  (name+"/SvB_ps_zh;  SvB Regressed P(Signal), P(ZH) is largest; Entries").c_str(), 100, 0, 1);
#    SvB_ps_zz = dir.make<TH1F>("SvB_ps_zz",  (name+"/SvB_ps_zz;  SvB Regressed P(Signal), P(ZZ) is largest; Entries").c_str(), 100, 0, 1);

#    FvT_q_score = dir.make<TH1F>("FvT_q_score", (name+"/FvT_q_score; FvT q_score (main pairing); Entries").c_str(), 100, 0, 1);
#    FvT_q_score_dR_min = dir.make<TH1F>("FvT_q_score_dR_min", (name+"/FvT_q_score; FvT q_score (min #DeltaR(j,j) pairing); Entries").c_str(), 100, 0, 1);
#    FvT_q_score_SvB_q_score_max = dir.make<TH1F>("FvT_q_score_SvB_q_score_max", (name+"/FvT_q_score; FvT q_score (max SvB q_score pairing); Entries").c_str(), 100, 0, 1);
#    SvB_q_score = dir.make<TH1F>("SvB_q_score", (name+"/SvB_q_score; SvB q_score; Entries").c_str(), 100, 0, 1);
#    SvB_q_score_FvT_q_score_max = dir.make<TH1F>("SvB_q_score_FvT_q_score_max", (name+"/SvB_q_score; SvB q_score (max FvT q_score pairing); Entries").c_str(), 100, 0, 1);
#    SvB_MA_q_score = dir.make<TH1F>("SvB_MA_q_score", (name+"/SvB_MA_q_score; SvB_MA q_score; Entries").c_str(), 100, 0, 1);

#    hT   = dir.make<TH1F>("hT", (name+"/hT; hT [GeV]; Entries").c_str(),  100,0,1000);


        fill += hist.add('nPVs', (101, -0.5, 100.5, ('PV.npvs', 'Number of Primary Vertices')))
        fill += hist.add('nPVsGood', (101, -0.5, 100.5, ('PV.npvsGood', 'Number of Good Primary Vertices')))

        fill += hist.add('FvT', (100, 0, 5, ('FvT.FvT', 'FvT reweight')))
        fill += hist.add('SvB_MA_ps', (100, 0, 1, ('SvB_MA.ps', 'SvB_MA Regressed P(Signal)')))
        fill += hist.add('SvB_ps', (100, 0, 1, ('SvB.ps', 'SvB Regressed P(Signal)')))
        fill += hist.add('quadJet_selected_dr', (50, 0, 5, ("quadJet_selected.dr",'Selected Diboson Candidate $\\Delta$R(d,d)')))
        fill += hist.add('quadJet_selected_dphi', (100, -3.2, 3.2, ("quadJet_selected.dphi",'Selected Diboson Candidate $\\Delta$R(d,d)')))
        fill += hist.add('quadJet_selected_deta', (100, -5, 5, ("quadJet_selected.deta",'Selected Diboson Candidate $\\Delta$R(d,d)')))

        for bb in self.signals:
            fill += hist.add(f'quadJet_selected_x{bb.upper()}', (100, 0, 10, (f"quadJet_selected.x{bb.upper()}", f'Selected Diboson Candidate X$_{bb.upper()}$')))
            fill += hist.add(f'SvB_ps_{bb}',    (100, 0, 1, (f'SvB.ps_{bb}', f"SvB Regressed P(Signal) $|$ P({bb.upper()}) is largest")))
            fill += hist.add(f'SvB_MA_ps_{bb}', (100, 0, 1, (f'SvB_MA.ps_{bb}', f"SvB MA Regressed P(Signal) $|$ P({bb.upper()}) is largest")))

        #
        # Jets
        #
        fill += Jet.plot(('selJets', 'Selected Jets'), 'selJet',skip=['deepjet_c'])
        fill += Jet.plot(('canJets', 'Higgs Candidate Jets'), 'canJet',skip=['deepjet_c'])
        fill += Jet.plot(('othJets', 'Other Jets'), 'notCanJet_coffea', skip=['deepjet_c'])
        
        for iJ in range(4):
            fill += Jet.plot((f'canJet{iJ}', f'Higgs Candidate Jets {iJ}'), f'canJet{iJ}', skip=['n','deepjet_c'])

        #
        #  v4j
        #
        fill += LorentzVector.plot_pair(('v4j', R'$HH_{4b}$'), 'v4j', skip=['n','dr','dphi','st'], bins = {'mass': (120, 0, 1200)})
        fill += LorentzVector.plot_pair(('leadSt', R'Lead Boson Candidate'), 'quadJet_selected_lead', skip=['n'])
        fill += LorentzVector.plot_pair(('sublSt', R'Subleading Boson Candidate'), 'quadJet_selected_subl', skip=['n'])
        #fill += LorentzVector.plot_pair(('p2j', R'Vector Boson Candidate Dijets'), 'p2jV')

        self.apply_puWeight   = (self.apply_puWeight  ) and isMC and (puWeight is not None)
        self.apply_prefire    = (self.apply_prefire   ) and isMC and ('L1PreFiringWeight' in event.fields) and (year!='UL18')
        self.apply_trigWeight = (self.apply_trigWeight) and isMC and ('trigWeight' in event.fields)
        self.apply_btagSF     = (self.apply_btagSF)     and isMC and (self.btagSF is not None)

        if isMC:
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            if self.btagSF is not None:
                btagSF = correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['btagSF'])['deepJet_shape']

            if self.apply_puWeight:
                puWeight = list(correctionlib.CorrectionSet.from_file(puWeight).values())[0]



        logging.debug(fname)
        logging.debug(f'{chunk}Process {nEvent} Events')

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split('/')[-1],'')
        event['FvT']    = NanoEventsFactory.from_root(f'{path}{"FvT_3bDvTMix4bDvT_v0_newSB.root" if "mix" in dataset else "FvT.root"}',    entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().FvT
        event['SvB']    = NanoEventsFactory.from_root(f'{path}{"SvB_newSBDef.root" if "mix" in dataset else "SvB.root"}',    entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB
        event['SvB_MA'] = NanoEventsFactory.from_root(f'{path}{"SvB_MA_newSBDef.root" if "mix" in dataset else "SvB_MA.root"}', entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB_MA

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
        setSvBVars("SvB", event)
        setSvBVars("SvB_MA", event)


        if isMC:
            self._cutFlow.fill("all",  event, allTag=True, wOverride = (lumi * xs * kFactor))
        else:
            self._cutFlow.fill("all",  event, allTag=True)

        #
        # Get trigger decisions
        #
        if year == 'UL16':
            event['passHLT'] = event.HLT.QuadJet45_TripleBTagCSV_p087 | event.HLT.DoubleJet90_Double30_TripleBTagCSV_p087 | event.HLT.DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6
        if year == 'UL17':
            event['passHLT'] = event.HLT.PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0 | event.HLT.DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33
        if year == 'UL18':
            event['passHLT'] = event.HLT.DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71 | event.HLT.PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5

        if not isMC and not 'mix' in dataset: # for data, apply trigger cut first thing, for MC, keep all events and apply trigger in cutflow and for plotting
            event = event[event.passHLT]

        if isMC:
            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)
            logging.debug(f"event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw) = {event.genWeight[0]} * ({lumi} * {xs} * {kFactor} / {genEventSumw}) = {event.weight[0]}\n")
            if self.apply_trigWeight: 
                event['weight'] = event.weight * event.trigWeight.Data
        else:
            event['weight'] = 1
            #logging.info(f"event['weight'] = {event.weight}")

        self._cutFlow.fill("passHLT",  event, allTag=True)


        #
        # METFilter
        #
        passMETFilter = np.ones(len(event), dtype=bool) if 'mix' in dataset else ( event.Flag.goodVertices & event.Flag.globalSuperTightHalo2016Filter & event.Flag.HBHENoiseFilter   & event.Flag.HBHENoiseIsoFilter & event.Flag.EcalDeadCellTriggerPrimitiveFilter & event.Flag.BadPFMuonFilter & event.Flag.eeBadScFilter)
        # passMETFilter *= event.Flag.EcalDeadCellTriggerPrimitiveFilter & event.Flag.BadPFMuonFilter                & event.Flag.BadPFMuonDzFilter & event.Flag.hfNoisyHitsFilter & event.Flag.eeBadScFilter
        if 'mix' not in dataset:
            if 'BadPFMuonDzFilter' in event.Flag.fields:
                passMETFilter = passMETFilter & event.Flag.BadPFMuonDzFilter
            if 'hfNoisyHitsFilter' in event.Flag.fields:
                passMETFilter = passMETFilter & event.Flag.hfNoisyHitsFilter
            if year == 'UL17' or year == 'UL18':
                passMETFilter = passMETFilter & event.Flag.ecalBadCalibFilter # in UL the name does not have "V2"
        #event['passMETFilter'] = passMETFilter


        #event = event[event.passMETFilter] # HACK
        self._cutFlow.fill("passMETFilter",  event, allTag=True)


        #
        # Calculate and apply Jet Energy Calibration   ## AGE: currently not applying to data and mixeddata
        #
        if isMC and juncWS is not None:
            jet_factory = init_jet_factory(juncWS)

            event['Jet', 'pt_raw']    = (1 - event.Jet.rawFactor) * event.Jet.pt
            event['Jet', 'mass_raw']  = (1 - event.Jet.rawFactor) * event.Jet.mass
            nominal_jet = event.Jet
            # nominal_jet['pt_raw']   = (1 - nominal_jet.rawFactor) * nominal_jet.pt
            # nominal_jet['mass_raw'] = (1 - nominal_jet.rawFactor) * nominal_jet.mass
            if isMC: nominal_jet['pt_gen']   = ak.values_astype(ak.fill_none(nominal_jet.matched_gen.pt, 0), np.float32)
            nominal_jet['rho']      = ak.broadcast_arrays(event.fixedGridRhoFastjetAll, nominal_jet.pt)[0]

            jec_cache = cachetools.Cache(np.inf)
            jet_variations = jet_factory.build(nominal_jet, lazy_cache=jec_cache)
            jet_tmp = jet_corrections( event.Jet, event.fixedGridRhoFastjetAll, jec_type=['L1L2L3Res'] )   ##### AGE: jsonpog+correctionlib but not final, that is why it is not used yet

        #
        # Loop over jet energy uncertainty variations running event selection, filling hists/cuflows independently for each jet calibration
        #
        for junc in self.juncVar:
            if junc != 'JES_Central':
                logging.debug(f'{chunk} running selection for {junc}')
                variation = '_'.join(junc.split('_')[:-1]).replace('YEAR', year)
                if 'JER' in junc: variation = variation.replace(f'_{year}','')
                direction = junc.split('_')[-1]
                # del event['Jet']
                event['Jet'] = jet_variations[variation, direction]

            event['Jet', 'calibration'] = event.Jet.pt/( 1 if 'data' in dataset else event.Jet.pt_raw )  ### AGE: I include the mix condition, I think it is wrong, to check later
            # if junc=='JES_Central':
            #     print(f'calibration nominal: \n{ak.mean(event.Jet.calibration)}')
            # else:
            #     print(f'calibration {variation} {direction}: \n{ak.mean(event.Jet.calibration)}')

            event['Jet', 'pileup'] = ((event.Jet.puId<0b110)&(event.Jet.pt<50)) | ((np.abs(event.Jet.eta)>2.4)&(event.Jet.pt<40))
            event['Jet', 'selected_loose'] = (event.Jet.pt>=20) & ~event.Jet.pileup
            event['Jet', 'selected'] = (event.Jet.pt>=40) & (np.abs(event.Jet.eta)<=2.4) & ~event.Jet.pileup
            event['nJet_selected'] = ak.sum(event.Jet.selected, axis=1)
            event['selJet'] = event.Jet[event.Jet.selected]


            selev = event[event.nJet_selected >= 4]
            self._cutFlow.fill("passJetMult",  selev, allTag=True)

            selev['Jet', 'tagged']       = selev.Jet.selected & (selev.Jet.btagDeepFlavB>=0.6)
            selev['Jet', 'tagged_loose'] = selev.Jet.selected & (selev.Jet.btagDeepFlavB>=0.3)
            selev['nJet_tagged']         = ak.num(selev.Jet[selev.Jet.tagged])
            selev['nJet_tagged_loose']   = ak.num(selev.Jet[selev.Jet.tagged_loose])

            fourTag  = (selev['nJet_tagged']       >= 4)
            threeTag = (selev['nJet_tagged_loose'] == 3) & (selev['nJet_selected'] >= 4)

            # check that coffea jet selection agrees with c++
            if junc == 'JES_Central':
                selev['issue'] = (threeTag!=selev.threeTag)|(fourTag!=selev.fourTag)
                if ak.any(selev.issue):
                    logging.warning(f'{chunk}WARNING: selected jets or fourtag calc not equal to picoAOD values')
                    logging.warning('nSelJets')
                    logging.warning(selev[selev.issue].nSelJets)
                    logging.warning(selev[selev.issue].nJet_selected)
                    logging.warning('fourTag')
                    logging.warning(selev.fourTag[selev.issue])
                    logging.warning(fourTag[selev.issue])

            selev[ 'fourTag']   =  fourTag
            selev['threeTag']   = threeTag * self.threeTag

            #selev['tag'] = ak.Array({'threeTag':selev.threeTag, 'fourTag':selev.fourTag})
            selev['passPreSel'] = selev.threeTag | selev.fourTag
            selev['tag'] = 0
            selev['tag'] = where(selev.passPreSel, (selev.fourTag, 4), (selev.threeTag, 3))

            #
            # Calculate and apply pileup weight, L1 prefiring weight
            #
            if self.apply_puWeight:
                for var in ['nominal', 'up', 'down']:
                    selev[f'PU_weight_{var}'] = puWeight.evaluate(selev.Pileup.nTrueInt.to_numpy(), var)
                selev['weight'] = selev.weight * selev.PU_weight_nominal

            if self.apply_prefire:
                selev['weight'] = selev.weight * selev.L1PreFiringWeight.Nom

            #
            # Calculate and apply btag scale factors
            #
            if isMC and btagSF is not None:

                #central = 'central'
                use_central = True
                btag_jes = []
                if junc != 'JES_Central':# and 'JER' not in junc:# and 'JES_Total' not in junc:
                    use_central = False
                    jes_or_jer = 'jer' if 'JER' in junc else 'jes'
                    btag_jes = [f'{direction}_{jes_or_jer}{variation.replace("JES_","").replace("Total","")}']
                cj, nj = ak.flatten(selev.selJet), ak.num(selev.selJet)
                hf, eta, pt, tag = np.array(cj.hadronFlavour), np.array(abs(cj.eta)), np.array(cj.pt), np.array(cj.btagDeepFlavB)

                cj_bl = selev.selJet[selev.selJet.hadronFlavour!=4]
                nj_bl = ak.num(cj_bl)
                cj_bl = ak.flatten(cj_bl)
                hf_bl, eta_bl, pt_bl, tag_bl = np.array(cj_bl.hadronFlavour), np.array(abs(cj_bl.eta)), np.array(cj_bl.pt), np.array(cj_bl.btagDeepFlavB)
                SF_bl= btagSF.evaluate('central', hf_bl, eta_bl, pt_bl, tag_bl)
                SF_bl = ak.unflatten(SF_bl, nj_bl)
                SF_bl = np.prod(SF_bl, axis=1)

                cj_c = selev.selJet[selev.selJet.hadronFlavour==4]
                nj_c = ak.num(cj_c)
                cj_c = ak.flatten(cj_c)
                hf_c, eta_c, pt_c, tag_c = np.array(cj_c.hadronFlavour), np.array(abs(cj_c.eta)), np.array(cj_c.pt), np.array(cj_c.btagDeepFlavB)
                SF_c= btagSF.evaluate('central', hf_c, eta_c, pt_c, tag_c)
                SF_c = ak.unflatten(SF_c, nj_c)
                SF_c = np.prod(SF_c, axis=1)

                for sf in self.btagVar+btag_jes:
                    if sf == 'central':
                        SF = btagSF.evaluate('central', hf, eta, pt, tag)
                        SF = ak.unflatten(SF, nj)

                        # hf = ak.unflatten(hf, nj)
                        # pt = ak.unflatten(pt, nj)
                        # eta = ak.unflatten(eta, nj)
                        # tag = ak.unflatten(tag, nj)
                        # for i in range(len(selev)):
                        #     for j in range(nj[i]):
                        #         print(f'jetPt/jetEta/jetTagScore/jetHadronFlavour/SF = {pt[i][j]}/{eta[i][j]}/{tag[i][j]}/{hf[i][j]}/{SF[i][j]}')
                        #     print(np.prod(SF[i]))
                        SF = np.prod(SF, axis=1)

                    if '_cf' in sf:
                        SF = btagSF.evaluate(sf, hf_c, eta_c, pt_c, tag_c)
                        SF = ak.unflatten(SF, nj_c)
                        SF = SF_bl * np.prod(SF, axis=1) # use central value for b,l jets
                    if '_hf' in sf or '_lf' in sf or '_jes' in sf:
                        SF = btagSF.evaluate(sf, hf_bl, eta_bl, pt_bl, tag_bl)
                        SF = ak.unflatten(SF, nj_bl)
                        SF = SF_c * np.prod(SF, axis=1) # use central value for charm jets


                    selev[f'btagSF_{sf}'] = SF * btagSF_norm
                    selev[f'weight_btagSF_{sf}'] = selev.weight * SF * btagSF_norm

                #
                #  Apply btag SF
                #
                if self.apply_btagSF:
                    selev['weight'] = selev[f'weight_btagSF_{"central" if use_central else btag_jes[0]}']

                self._cutFlow.fill("passJetMult_btagSF",  selev, allTag=True)


            # for i in range(len(selev)):
            #     print(selev.event[i], selev.btagSF_central[i])


            #
            # Preselection: keep only three or four tag events
            #
            selev = selev[selev.passPreSel]


            #
            # Build and select boson candidate jets with bRegCorr applied
            #
            sorted_idx = ak.argsort(selev.Jet.btagDeepFlavB * selev.Jet.selected, axis=1, ascending=False)
            canJet_idx = sorted_idx[:,0:4]
            notCanJet_idx = sorted_idx[:,4:]
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

            # pt sort canJets
            canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
            selev['canJet'] = canJet

            #
            #  Should be a better way to do this...
            # 
            selev['canJet0'] = canJet[:,0]
            selev['canJet1'] = canJet[:,1]
            selev['canJet2'] = canJet[:,2]
            selev['canJet3'] = canJet[:,3]
            
            selev['v4j'] = canJet.sum(axis=1)
            #selev['v4j', 'n'] = 1
            #print(selev.v4j.n)
            # selev['Jet', 'canJet'] = False
            # selev.Jet.canJet.Fill(canJet_idx, True)
            notCanJet = selev.Jet[notCanJet_idx]
            notCanJet = notCanJet[notCanJet.selected_loose]
            notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]
            notCanJet['isSelJet'] = 1*((notCanJet.pt>40) & (np.abs(notCanJet.eta)<2.4)) # should have been defined as notCanJet.pt>=40, too late to fix this now...
            selev['notCanJet_coffea'] = notCanJet
            selev['nNotCanJet'] = ak.num(selev.notCanJet_coffea)

            # if junc=='JES_Central':
            #     print(f'{ak.mean(canJet.calibration)} (canJets)')
            # else:
            #     print(f'{ak.mean(canJet.calibration)} (canJets)')
            # print(canJet_idx[0])
            # print(selev[0].Jet[canJet_idx[0]].pt)
            # print(selev[0].Jet[canJet_idx[0]].bRegCorr)
            # print(selev[0].Jet[canJet_idx[0]].calibration)

            if self.threeTag:
                #
                # calculate pseudoTagWeight for threeTag events
                #
                selev['Jet_untagged_loose'] = selev.Jet[selev.Jet.selected & ~selev.Jet.tagged_loose]
                nJet_pseudotagged = np.zeros(len(selev), dtype=int)
                pseudoTagWeight = np.ones(len(selev))
                pseudoTagWeight[selev.threeTag], nJet_pseudotagged[selev.threeTag] = self.JCM(selev[selev.threeTag]['Jet_untagged_loose'])
                selev['nJet_pseudotagged'] = nJet_pseudotagged

                # check that pseudoTagWeight calculation agrees with c++
                if junc == 'JES_Central':
                    selev.issue = (abs(selev.pseudoTagWeight - pseudoTagWeight)/selev.pseudoTagWeight > 0.0001) & (pseudoTagWeight!=1)
                    if ak.any(selev.issue):
                        logging.warning(f'{chunk}WARNING: python pseudotag calc not equal to c++ calc')
                        logging.warning(f'{chunk}Issues:',ak.sum(selev.issue),'of',ak.sum(selev.threeTag))

                # add pseudoTagWeight to event
                selev['pseudoTagWeight'] = pseudoTagWeight
                
                #logging.info(f'pseudoTagWeight: {selev.pseudoTagWeight}')

                #
                # apply pseudoTagWeight and FvT to threeTag events
                #
                if self.doReweight:
                    selev['weight'] = where(selev.passPreSel, (selev.threeTag, selev.weight * selev.pseudoTagWeight * selev.FvT.FvT), (selev.fourTag, selev.weight))
                else:
                    selev['weight'] = where(selev.passPreSel, (selev.threeTag, selev.weight * selev.pseudoTagWeight), (selev.fourTag, selev.weight))

            #
            # CutFlow
            #
            self._cutFlow.fill("passPreSel",  selev)

            #
            # Build diJets, indexed by diJet[event,pairing,0/1]
            #
            canJet = selev['canJet']
            pairing = [([0,2],[0,1],[0,1]),
                       ([1,3],[2,3],[3,2])]
            diJet       = canJet[:,pairing[0]]     +   canJet[:,pairing[1]]
            diJet['st'] = canJet[:,pairing[0]].pt  +   canJet[:,pairing[1]].pt
            diJet['dr'] = canJet[:,pairing[0]].delta_r(canJet[:,pairing[1]])
            diJet['dphi'] = canJet[:,pairing[0]].delta_phi(canJet[:,pairing[1]])
            diJet['lead'] = canJet[:,pairing[0]]
            diJet['subl'] = canJet[:,pairing[1]]
            # Sort diJets within views to be lead st, subl st
            diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
            # Now indexed by diJet[event,pairing,lead/subl st]

            # Compute diJetMass cut with independent min/max for lead/subl
            minDiJetMass = np.array([[[ 52, 50]]])
            maxDiJetMass = np.array([[[180,173]]])
            diJet['passDiJetMass'] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

            # Compute MDRs
            min_m4j_scale = np.array([[ 360, 235]])
            min_dr_offset = np.array([[-0.5, 0.0]])
            max_m4j_scale = np.array([[ 650, 650]])
            max_dr_offset = np.array([[ 0.5, 0.7]])
            max_dr        = np.array([[ 1.5, 1.5]])
            m4j = np.repeat(np.reshape(np.array(selev['v4j'].mass), (-1,1,1)), 2, axis=2)
            diJet['passMDR'] = (min_m4j_scale/m4j + min_dr_offset < diJet.dr) & (diJet.dr < np.maximum(max_m4j_scale/m4j + max_dr_offset, max_dr))

            # Compute consistency of diJet masses with boson masses
            mZ =  91.0
            mH = 125.0
            st_bias = np.array([[[1.02, 0.98]]])
            cZ = mZ * st_bias
            cH = mH * st_bias

            diJet['xZ'] = (diJet.mass - cZ)/(0.1*diJet.mass)
            diJet['xH'] = (diJet.mass - cH)/(0.1*diJet.mass)

            #
            # Build quadJets
            #
            quadJet = ak.zip({'lead': diJet[:,:,0],
                              'subl': diJet[:,:,1],
                              'passDiJetMass': ak.all(diJet.passDiJetMass, axis=2),
                              'random': np.random.uniform(low=0.1, high=0.9, size=(diJet.__len__(), 3))
                          })#, with_name='quadJet')
            quadJet['dr']   = quadJet['lead'].delta_r(quadJet['subl'])
            quadJet['dphi'] = quadJet['lead'].delta_phi(quadJet['subl'])
            quadJet['deta'] = quadJet['lead'].eta - quadJet['subl'].eta
            quadJet['SvB_q_score'] = np.concatenate((np.reshape(np.array(selev.SvB.q_1234), (-1,1)),
                                                     np.reshape(np.array(selev.SvB.q_1324), (-1,1)),
                                                     np.reshape(np.array(selev.SvB.q_1423), (-1,1))), axis=1)
            quadJet['SvB_MA_q_score'] = np.concatenate((np.reshape(np.array(selev.SvB_MA.q_1234), (-1,1)),
                                                        np.reshape(np.array(selev.SvB_MA.q_1324), (-1,1)),
                                                        np.reshape(np.array(selev.SvB_MA.q_1423), (-1,1))), axis=1)

            # Compute Signal Regions
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

            # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
            quadJet['rank'] = 10*quadJet.passDiJetMass + quadJet.lead.passMDR + quadJet.subl.passMDR + quadJet.random
            quadJet['selected'] = quadJet.rank == np.max(quadJet.rank, axis=1)

            selev[  'diJet'] =   diJet
            selev['quadJet'] = quadJet
            selev['quadJet_selected'] = quadJet[quadJet.selected][:,0]
            selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass,axis=1)

            # FIX ME  (Better way to do this
            selev['quadJet_selected_lead'] = selev['quadJet_selected'].lead
            selev['quadJet_selected_subl'] = selev['quadJet_selected'].subl

            selev['region'] = selev['quadJet_selected'].SR * 0b10 + selev['quadJet_selected'].SB * 0b01
            selev['passSvB'] = (selev['SvB_MA'].ps > 0.95)
            selev['failSvB'] = (selev['SvB_MA'].ps < 0.05)

            # selev.issue = (selev.leadStM<0) | (selev.sublStM<0)
            # if ak.any(selev.issue):
            #     print(f'{chunk}WARNING: Negative diJet masses in picoAOD variables generated by the c++')
            #     issue = selev[selev.issue]
            #     print(f'{chunk}{len(issue)} events with issues')
            #     print(f'{chunk}c++ values:',issue.passDiJetMass, issue.leadStM,issue.sublStM)
            #     print(f'{chunk}py  values:',issue.quadJet_selected.passDiJetMass, issue.quadJet_selected.lead.mass, issue.quadJet_selected.subl.mass)

            # if junc == 'JES_Central':
            #     selev.issue = selev.passDijetMass != selev['quadJet_selected'].passDiJetMass
            #     selev.issue = selev.issue & ~((selev.leadStM<0) | (selev.sublStM<0))
            #     if ak.any(selev.issue):
            #         print(f'{chunk}WARNING: passDiJetMass calc not equal to picoAOD value')
            #         issue = selev[selev.issue]
            #         print(f'{chunk}{len(issue)} events with issues')
            #         print(f'{chunk}c++ values:',issue.passDijetMass, issue.leadStM,issue.sublStM)
            #         print(f'{chunk}py  values:',issue.quadJet_selected.passDiJetMass, issue.quadJet_selected.lead.mass, issue.quadJet_selected.subl.mass)

            #
            # Blind data in fourTag SR
            #
            if not (isMC or 'mixed' in dataset) and self.blind:
                selev = selev[~(selev['quadJet_selected'].SR & selev.fourTag)]

            if self.classifier_SvB is not None:
                self.compute_SvB(selev, junc=junc)

            #
            # fill histograms
            #
            self._cutFlow.fill("passDiJetMass",  selev[selev.passDiJetMass])
            self._cutFlow.fill("SR",  selev[(selev.passDiJetMass & selev['quadJet_selected'].SR)])
            self._cutFlow.fill("SB",  selev[(selev.passDiJetMass & selev['quadJet_selected'].SB)])
            self._cutFlow.fill("passSvB",  selev[selev.passSvB])
            self._cutFlow.fill("failSvB",  selev[selev.failSvB])

            #fill.cache(selev)
            fill(selev)

            garbage = gc.collect()
            # print('Garbage:',garbage)


        # Done
        elapsed = time.time() - tstart
        logging.debug(f'{chunk}{nEvent/elapsed:,.0f} events/s')

        self._cutFlow.addOutput(processOutput, event.metadata['dataset'])

        return hist.output | processOutput


    def compute_SvB(self, event, junc='JES_Central'):
        n = len(event)

        j = torch.zeros(n, 4, 4)
        j[:,0,:] = torch.tensor( event.canJet.pt   )
        j[:,1,:] = torch.tensor( event.canJet.eta  )
        j[:,2,:] = torch.tensor( event.canJet.phi  )
        j[:,3,:] = torch.tensor( event.canJet.mass )

        o = torch.zeros(n, 5, 8)
        o[:,0,:] = torch.tensor( ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.pt,       target=8, clip=True)),  0) )
        o[:,1,:] = torch.tensor( ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.eta,      target=8, clip=True)),  0) )
        o[:,2,:] = torch.tensor( ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.phi,      target=8, clip=True)),  0) )
        o[:,3,:] = torch.tensor( ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.mass,     target=8, clip=True)),  0) )
        o[:,4,:] = torch.tensor( ak.fill_none(ak.to_regular(ak.pad_none(event.notCanJet_coffea.isSelJet, target=8, clip=True)), -1) )

        a = torch.zeros(n, 4)
        a[:,0] =        float( event.metadata['year'][3] )
        a[:,1] = torch.tensor( event.nJet_selected )
        a[:,2] = torch.tensor( event.xW )
        a[:,3] = torch.tensor( event.xbW )

        e = torch.tensor(event.event)%3

        for classifier in ['SvB', 'SvB_MA']:
            if classifier == 'SvB':
                c_logits, q_logits = self.classifier_SvB(j, o, a, e)
            if classifier == 'SvB_MA':
                c_logits, q_logits = self.classifier_SvB_MA(j, o, a, e)

            c_score, q_score = F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy()

            # classes = [mj,tt,zz,zh,hh]
            SvB = ak.zip({'pmj': c_score[:,0],
                          'ptt': c_score[:,1],
                          'pzz': c_score[:,2],
                          'pzh': c_score[:,3],
                          'phh': c_score[:,4],
                          'q_1234': q_score[:,0],
                          'q_1324': q_score[:,1],
                          'q_1423': q_score[:,2],
                      })
            SvB['ps'] = SvB.pzz + SvB.pzh + SvB.phh
            SvB['passMinPs'] = (SvB.pzz>0.01) | (SvB.pzh>0.01) | (SvB.phh>0.01)
            SvB['zz'] = (SvB.pzz >  SvB.pzh) & (SvB.pzz >  SvB.phh)
            SvB['zh'] = (SvB.pzh >  SvB.pzz) & (SvB.pzh >  SvB.phh)
            SvB['hh'] = (SvB.phh >= SvB.pzz) & (SvB.phh >= SvB.pzh)


            if junc == 'JES_Central':
                error = ~np.isclose(event[classifier].ps, SvB.ps, atol=1e-5, rtol=1e-3)
                if np.any(error):
                    delta = np.abs(event[classifier].ps - SvB.ps)
                    worst = np.max(delta) == delta #np.argmax(np.abs(delta))
                    worst_event = event[worst][0]
                    logging.warning(f'WARNING: Calculated {classifier} does not agree within tolerance for some events ({np.sum(error)}/{len(error)})', delta[worst])
                    logging.warning('----------')
                    for field in event[classifier].fields:
                          logging.warning(field, worst_event[classifier][field])
                    logging.warning('----------')
                    for field in SvB.fields:
                        logging.warning( f'{field}, {SvB[worst][field]}')

            # del event[classifier]
            event[classifier] = SvB



    def postprocess(self, accumulator):
        #return accumulator
        ...

