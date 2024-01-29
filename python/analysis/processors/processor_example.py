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

from coffea.nanoevents.methods import vector
from coffea import processor

from base_class.hist import Collection, Fill
from base_class.hist import H, Template
from base_class.physics.object import LorentzVector, Jet, Muon, Elec

from analysis.helpers.FriendTreeSchema import FriendTreeSchema

from functools import partial
from multiprocessing import Pool

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
    def __init__(self, ):

    def process(self, event):
        tstart = time.time()
        dataset = event.metadata['dataset']

        processOutput = {}
        processOutput['nEvent'] = {}
        processOutput['nEvent'][event.metadata['dataset']] = nEvent


        #
        # general event weights
        #
        if isMC:
            # genWeight
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])

            event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw)
            logging.debug(f"event['weight'] = event.genWeight * (lumi * xs * kFactor / genEventSumw) = {event.genWeight[0]} * ({lumi} * {xs} * {kFactor} / {genEventSumw}) = {event.weight[0]}\n")

            # trigger Weight (to be updated)
            #event['weight'] = event.weight * event.trigWeight.Data

            #puWeight
            puWeight = list(correctionlib.CorrectionSet.from_file(self.corrections_metadata[year]['PU']).values())[0]
            for var in ['nominal', 'up', 'down']:
                event[f'PU_weight_{var}'] = puWeight.evaluate(event.Pileup.nTrueInt.to_numpy(), var)
            event['weight'] = event.weight * event.PU_weight_nominal
        else:
            event['weight'] = 1

        # L1 prefiring weight
        if isMC & ('L1PreFiringWeight' in event.fields):   #### AGE: this should be temprorary (field exists in UL)
            event['weight'] = event.weight * event.L1PreFiringWeight.Nom
        # logging.info(f"event['weight'] = {event.weight}")

        #
        # Event selection (function only adds flags, not remove events)
        #
        events = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year]  )

        #
        # Filtering object and event selection
        #
        selev = event[ event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult ]


        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f'{chunk}{nEvent/elapsed:,.0f} events/s')


        return processOutput



    def postprocess(self, accumulator):
        ...
