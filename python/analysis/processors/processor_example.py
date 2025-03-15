import time
import gc
import numpy as np
import uproot
import correctionlib
import yaml
import warnings

from coffea.nanoevents import NanoAODSchema
from coffea import processor

from base_class.hist import Collection, Fill
from base_class.physics.object import Jet

from analysis.helpers.correctionFunctions import btagVariations
from analysis.helpers.correctionFunctions import btagSF_norm as btagSF_norm_file
from analysis.helpers.cutflow import cutFlow


from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.common import apply_btag_sf
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b
import logging


#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(self, JCM='', corrections_metadata='analysis/metadata/corrections.yml'):
        self.histCuts = ['passPreSel']
        self.tags = ['threeTag', 'fourTag']
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
        self._cutFlow = cutFlow()

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split('/')[-1], '')
        #event['FvT']    = NanoEventsFactory.from_root(f'{path}{"FvT_3bDvTMix4bDvT_v0_newSB.root" if "mix" in dataset else "FvT.root"}',
        #                                              entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().FvT

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
        self._cutFlow.fill("passJetMult",  event[ event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult ], allTag=True)

        #
        # Filtering object and event selection
        #
        selev = event[ event.lumimask & event.passNoiseFilter & event.passHLT & event.passJetMult ]

        #
        # Calculate and apply btag scale factors
        #
        if isMC:
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
        #  Add here your code
        #


        #
        # Hists
        #
        fill = Fill(process=processName, year=year, weight='weight')

        hist = Collection(process = [processName],
                          year    = [year],
                          tag     = [3, 4, 0],    # 3 / 4/ Other
                          **dict((s, ...) for s in self.histCuts))

        fill += hist.add('nPVs',     (101, -0.5, 100.5, ('PV.npvs',     'Number of Primary Vertices')))
        fill += hist.add('nPVsGood', (101, -0.5, 100.5, ('PV.npvsGood', 'Number of Good Primary Vertices')))

        fill += Jet.plot(('selJets', 'Selected Jets'),        'selJet',           skip=['deepjet_c'])
        fill += Jet.plot(('tagJets', 'Tag Jets'),             'tagJet',           skip=['deepjet_c'])

        #
        # fill histograms
        #
        # fill.cache(selev)
        fill(selev)

        #
        # CutFlow
        #
        self._cutFlow.fill("passPreSel", selev)


        garbage = gc.collect()
        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f'{chunk}{nEvent/elapsed:,.0f} events/s')


        self._cutFlow.addOutput(processOutput, event.metadata['dataset'])

        return hist.output | processOutput


    def postprocess(self, accumulator):
        ...
