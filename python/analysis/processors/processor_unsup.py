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
import yaml
import warnings

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import vector
from coffea import processor

from base_class.hist import Collection, Fill
from base_class.hist import H, Template
from base_class.physics.object import LorentzVector, Jet, Muon, Elec

from analysis.helpers.FriendTreeSchema import FriendTreeSchema
from analysis.helpers.correctionFunctions import btagVariations
from analysis.helpers.correctionFunctions import btagSF_norm as btagSF_norm_file
from analysis.helpers.cutflow import cutFlow

from functools import partial
from multiprocessing import Pool

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

from base_class.hist import H, Template


class analysis(processor.ProcessorABC):
    def __init__(self, JCM='', threeTag = True, corrections_metadata='analysis/metadata/corrections.yml'):
        logging.debug('\nInitialize Analysis Processor')
        self.cutFlowCuts = ["all", "passHLT", "passNoiseFilter", "passJetMult", "passJetMult_btagSF", "passPreSel"]
        self.histCuts = ['passPreSel']
        self.tags = ['threeTag', 'fourTag'] if threeTag else ['fourTag']
        self.JCM = jetCombinatoricModel(JCM)
        self.btagVar = btagVariations(systematics=True)  #### AGE: these two need to be review later
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, 'r'))

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

        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = nEvent
        self._cutFlow = cutFlow(self.cutFlowCuts)

        ########################################
        ###### Reading SvB friend trees ########
        ########################################

        path = fname.replace(fname.split('store/user')[-1], '')
    
        if fname.find('picoAOD_3b_wJCM_newSBDef') != -1:
            fname_w3to4 = f"/smurthy/condor/unsupervised4b/randPair/w3to4hist/data20{year[-2:]}_picoAOD_3b_wJCM_newSBDef_w3to4_hist.root"
            fname_wDtoM = f"/smurthy/condor/unsupervised4b/randPair/wDtoMwJMC/data20{year[-2:]}_picoAOD_3b_wJCM_newSBDef_wDtoM.root"
            event['w3to4'] = NanoEventsFactory.from_root(f'{path}{fname_w3to4}', 
                            entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().w3to4.w3to4
            
            event['wDtoM'] = NanoEventsFactory.from_root(f'{path}{fname_wDtoM}', 
                            entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().wDtoM.wDtoM

            #### event['w3to4', 'frac_err'] = event['w3to4'].std / event['w3to4'].w3to4

            # if not ak.all(event.w3to4.event == event.event):
            #     logging.error('ERROR: w3to4 events do not match events ttree')
            #     return
            
            # if not ak.all(event.wDtoM.event == event.event):
            #     logging.error('ERROR: wDtoM events do not match events ttree')
            #     return
    


        ##############################################
        ### general event weights
        if isMC:
            ### genWeight
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)
            logging.debug(f"event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw) = {event.genWeight[0]} * ({lumi} * {xs} * {kFactor} / {genEventSumw}) = {event.weight[0]}\n")

            ### trigger Weight (to be updated)
            ###event['weight'] = event.weight * event.trigWeight.Data

            ###puWeight
            puWeight = list(correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['PU']).values())[0]
            for var in ['nominal', 'up', 'down']:
                event[f'PU_weight_{var}'] = puWeight.evaluate(event.Pileup.nTrueInt.to_numpy(), var)
            event['weight'] = event.weight * event.PU_weight_nominal

            ### L1 prefiring weight
            if ('L1PreFiringWeight' in event.fields):   #### AGE: this should be temprorary (field exists in UL)
                event['weight'] = event.weight * event.L1PreFiringWeight.Nom
        else:
            event['weight'] = 1
        #    if True:
        

        logging.debug(f"event['weight'] = {event.weight}")

        
        ### Event selection (function only adds flags, not remove events)
        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )

        self._cutFlow.fill("all",  event[event.lumimask], allTag=True)
        self._cutFlow.fill("passNoiseFilter",  event[ event.lumimask & event.passNoiseFilter], allTag=True)
        self._cutFlow.fill("passHLT",  event[ event.lumimask & event.passNoiseFilter & event.passHLT], allTag=True)

        ### Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year]  )
        self._cutFlow.fill("passJetMult",  event[ event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult ], allTag=True)

        ### Filtering object and event selection
        selev = event[ event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult ]

        
        ##### Calculate and apply btag scale factors
        if isMC:
            btagSF = correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['btagSF'])['deepJet_shape']
            selev['weight'] = apply_btag_sf(selev, selev.selJet,
                                            correction_file=self.corrections_metadata[year]['btagSF'],
                                            btag_var=self.btagVar,
                                            btagSF_norm=btagSF_norm_file(dataset),
                                            weight=selev.weight )

            self._cutFlow.fill("passJetMult_btagSF",  selev, allTag=True)

        
        ### Preselection: keep only three or four tag events
        
        selev = selev[selev.passPreSel]
        if fname.find('picoAOD_3b_wJCM_newSBDef') != -1:
            selev['weight_wDtoM'] = selev.weight * selev.wDtoM
            selev['weight_wDtoM_w3to4'] = selev.weight_wDtoM * selev.w3to4
        ############################################
        ############## Unsup 4b code ###############
        ############################################

        #### Calculate hT (scalar sum of jet pts)
    
        selev['hT']          = ak.sum(selev.Jet[selev.Jet.selected_loose].pt, axis=1)
        selev['hT_selected'] = ak.sum(selev.Jet[selev.Jet.selected      ].pt, axis=1)

        
        ### Build and select boson candidate jets with bRegCorr applied
        
        sorted_idx = ak.argsort(selev.Jet.btagDeepFlavB * selev.Jet.selected, axis=1, ascending=False)
        canJet_idx    = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]
        canJet = selev.Jet[canJet_idx]

        ### apply bJES to canJets
        canJet = canJet * canJet.bRegCorr
        canJet['bRegCorr'] = selev.Jet.bRegCorr[canJet_idx]
        canJet['btagDeepFlavB'] = selev.Jet.btagDeepFlavB[canJet_idx]
        canJet['puId'] = selev.Jet.puId[canJet_idx]
        canJet['jetId'] = selev.Jet.puId[canJet_idx]
        if isMC:
            canJet['hadronFlavour'] = selev.Jet.hadronFlavour[canJet_idx]
        canJet['calibration'] = selev.Jet.calibration[canJet_idx]

        ### pt sort canJets    
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        selev['canJet'] = canJet

        ###  Should be a better way to do this...        
        selev['canJet0'] = canJet[:, 0]
        selev['canJet1'] = canJet[:, 1]
        selev['canJet2'] = canJet[:, 2]
        selev['canJet3'] = canJet[:, 3]
        selev['v4j'] = canJet.sum(axis=1)
        selev['m4j'] = selev.v4j.mass
        
        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        notCanJet['isSelJet'] = 1 * ((notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4))     # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev['notCanJet_coffea'] = notCanJet
        selev['nNotCanJet'] = ak.num(selev.notCanJet_coffea)

    
        ### Build diJets, indexed by diJet[event,pairing,0/1]
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
        # # Now indexed by diJet[event,pairing,lead/subl st]

        # Compute diJetMass cut with independent min/max for lead/subl
        minDiJetMass = np.array([[[ 0,  0]]])
        maxDiJetMass = np.array([[[1000, 1000]]])
        diJet['passDiJetMass'] = (minDiJetMass < diJet.mass) & (diJet.mass < maxDiJetMass)

        ### Build quadJets

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

        
        # #
        # # Compute Signal Regions
        # #
        # quadJet['xZZ'] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
        # quadJet['xHH'] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
        # quadJet['xZH'] = np.sqrt(np.minimum(quadJet.lead.xH**2 + quadJet.subl.xZ**2,
        #                                     quadJet.lead.xZ**2 + quadJet.subl.xH**2))

        # max_xZZ = 2.6
        # max_xZH = 1.9
        # max_xHH = 1.9
        # quadJet['ZZSR'] = quadJet.xZZ < max_xZZ
        # quadJet['ZHSR'] = quadJet.xZH < max_xZH
        # quadJet['HHSR'] = quadJet.xHH < max_xHH
        # quadJet['SR'] = quadJet.ZZSR | quadJet.ZHSR | quadJet.HHSR
        # quadJet['SB'] = quadJet.passDiJetMass & ~quadJet.SR

        # #
        # #  Build the close dR and other quadjets
        # #    (There is Probably a better way to do this ...
        # #
        # arg_min_close_dr = np.argmin(quadJet.close.dr, axis=1)
        # arg_min_close_dr = arg_min_close_dr.to_numpy()
        # selev['quadJet_min_dr'] = quadJet[np.array(range(len(quadJet))), arg_min_close_dr]

        # #
        # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
        #
        quadJet['rank'] = quadJet.random
        quadJet['selected'] = quadJet.rank == np.max(quadJet.rank, axis=1)

        selev['diJet'] = diJet
        selev['quadJet'] = quadJet
        selev['quadJet_selected'] = quadJet[quadJet.selected][:, 0]
        selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)
        selev['leadStM'] = selev.quadJet_selected.lead.mass
        selev['sublStM'] = selev.quadJet_selected.subl.mass

        # selev['region'] = selev['quadJet_selected'].SR * 0b10 + selev['quadJet_selected'].SB * 0b01
        # selev['passSvB'] = (selev['SvB_MA'].ps > 0.80)
        # selev['failSvB'] = (selev['SvB_MA'].ps < 0.05)

        #
        # Blind data in fourTag SR
        #
        # if not (isMC or 'mixed' in dataset) and self.blind:
        #     selev = selev[~(selev['quadJet_selected'].SR & selev.fourTag)]

        # #
        # # Compute Signal Regions
        # #
        


        # # pick quadJet at random
        # #
    

        

        # #
        # # Blind data in fourTag SR
        # #
        # if not (isMC or 'mixed' in dataset) and self.blind:
        #     selev = selev[~(selev['quadJet_selected'].SR & selev.fourTag)]


        #################################################################
        #
        ### Hists
        #

        fill = Fill(process=processName, year=year, weight='weight')

        hist = Collection(process = [processName],
                          year    = [year],
                          tag     = [3, 4, 0],    # 3 / 4/ Other
                          **dict((s, ...) for s in self.histCuts))

        fill += hist.add('nPVs',     (101, -0.5, 100.5, ('PV.npvs',     'Number of Primary Vertices')))
        fill += hist.add('nPVsGood', (101, -0.5, 100.5, ('PV.npvsGood', 'Number of Good Primary Vertices')))
        fill += hist.add('m4j', (100, 0, 1000, ('m4j', 'm4j data')))
        
        if fname.find('picoAOD_3b_wJCM_newSBDef') != -1:
            fill += hist.add('m4j_wDtoM', (100, 0, 1000, ('m4j', 'm4j multijet')), weight="weight_wDtoM")
            fill += hist.add('m4j_bkg', (100, 0, 1000, ('m4j', 'm4j background')), weight="weight_wDtoM_w3to4")
            # fill += hist.add('leadStM_bkg', (100, 0, 1000, ('leadStM', 'leadStM background')), weight="weight_wDtoM_w3to4")
            # fill += hist.add('sublStM_bkg', (100, 0, 1000, ('sublStM', 'sublStM background')), weight="weight_wDtoM_w3to4")
            # fill += hist.add('nJet_selected', (16, 0, 15, ('nJet_selected', 'nJet_selected background')), weight="weight_wDtoM_w3to4")
        
        fill += Jet.plot(('selJets', 'Selected Jets'),        'selJet',           skip=['deepjet_c'])
        fill += Jet.plot(('tagJets', 'Tag Jets'),             'tagJet',           skip=['deepjet_c'])

        
        ### fill histograms ###
        # fill.cache(selev)
        fill(selev)

        
        ### CutFlow ###
        self._cutFlow.fill("passPreSel", selev)
        self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
        self._cutFlow.fill("passThreeTag", selev[selev.threeTag])
        self._cutFlow.fill("passFourTag", selev[selev.fourTag])


        garbage = gc.collect()
    
        ### Done ###
        elapsed = time.time() - tstart
        logging.debug(f'{chunk}{nEvent/elapsed:,.0f} events/s')


        self._cutFlow.addOutput(processOutput, event.metadata['dataset'])

        return hist.output | processOutput


    def postprocess(self, accumulator):
        ...
