import logging
import warnings
import awkward as ak
import yaml
import numpy as np
from analysis.helpers.common import init_jet_factory
from base_class.trigger_emulator.TrigEmulatorTool   import TrigEmulatorTool
from analysis.helpers.selection_basic_4b import (
    apply_event_selection_4b,
    apply_object_selection_4b,
    create_cand_jet_dijet_quadjet,
)
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema
from analysis.helpers.dump_friendtrees import dump_trigger_weight


#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        make_classifier_input: str = None,
        corrections_metadata: str ="analysis/metadata/corrections.yml",
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, "r"))
        self.make_classifier_input = make_classifier_input

        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
        ]


    def process(self, event):

        self.dataset = event.metadata['dataset']
        self.year    = event.metadata['year']
        self.processName = event.metadata['processName']
        self.isMC    = False if "data" in self.processName else True

        self.isMixedData    = not (self.dataset.find("mix_v") == -1)
        self.isDataForMixed = not (self.dataset.find("data_3b_for_mixed") == -1)
        self.isTTForMixed   = not (self.dataset.find("TTTo") == -1) and not ( self.dataset.find("_for_mixed") == -1 )

        if self.isMixedData:
            self.isMC = False

        #
        #  Nominal config (...what we would do for data)
        #
        self.cut_on_lumimask         = True
        self.do_lepton_jet_cleaning  = True

        if self.isMC:
            self.cut_on_lumimask     = False
            self.do_jet_calibration  = True


        self.nEvent = len(event)

        #
        # Event selection
        #
        event = apply_event_selection_4b( event, self.corrections_metadata[self.year], cut_on_lumimask=self.cut_on_lumimask)

        #
        # Calculate and apply Jet Energy Calibration
        #
        juncWS = [ self.corrections_metadata[self.year]["JERC"][0].replace("STEP", istep)
                    for istep in ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"] ] + self.corrections_metadata[self.year]["JERC"][2:]

        jets = init_jet_factory(juncWS, event, self.isMC)
        event["Jet"] = jets

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, self.corrections_metadata[self.year],
                                           doLeptonRemoval=self.do_lepton_jet_cleaning )

        create_cand_jet_dijet_quadjet( event, event.event,
                                      isMC = self.isMC,
                                      apply_FvT=False,
                                      apply_boosted_veto=False,
                                      run_SvB=False,
                                      run_systematics=False,
                                      classifier_SvB=None,
                                      classifier_SvB_MA=None,
                                      )

        year_label = self.corrections_metadata[self.year]['year_label'].replace("UL", "20").split("_")[0]
        emulator_data = TrigEmulatorTool("Test", year=year_label, nToys=100)
        emulator_mc   = TrigEmulatorTool("Test", year=year_label, nToys=100, useMCTurnOns=True)
        event['trigWeight'] = {}
        event['trigWeight', "Data"] = ak.Array([ emulator_data.GetWeightOR(selJet_pt, tagJet_pt, hT_trigger) for selJet_pt, tagJet_pt, hT_trigger in zip(event.selJet.pt, event.canJet.pt, event.hT_trigger) ])
        event['trigWeight', 'MC' ] = ak.Array([ emulator_mc.GetWeightOR(selJet_pt, tagJet_pt, hT_trigger) for selJet_pt, tagJet_pt, hT_trigger in zip(event.selJet.pt, event.canJet.pt, event.hT_trigger) ])

        logging.debug(f"trigger weight data: {event['trigWeight'].Data}")
        logging.debug(f"trigger weight mc: {event['trigWeight'].MC}")

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        allcuts = [ 'lumimask', 'passNoiseFilter' ]

        friends = {}

        friends["friends"] = dump_trigger_weight( event, self.make_classifier_input,
                                                 "trigWeight",
                                                  selections.all(*allcuts))

        return friends

    def postprocess(self, accumulator):
        return accumulator
