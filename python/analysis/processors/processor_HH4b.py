import logging
import warnings

import awkward as ak
import numpy as np
import yaml, json
import copy

from analysis.helpers.processor_config import processor_config
from analysis.helpers.common import apply_jerc_corrections, update_events

from analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms
)
from analysis.helpers.event_weights import (
    add_weights,
    add_pseudotagweights,
    add_btagweights,
)
from analysis.helpers.cutflow import cutFlow
from analysis.helpers.FriendTreeSchema import FriendTreeSchema
from analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_SvB
from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.selection_basic_4b import (
    apply_event_selection_4b,
    apply_object_selection_4b,
    create_cand_jet_dijet_quadjet,
)
from analysis.helpers.topCandReconstruction import (
    buildTop,
    dumpTopCandidateTestVectors,
    find_tops,
    find_tops_slow,
    adding_top_reco_to_event,
)
from base_class.root import Chunk, TreeReader, Friend
from base_class.utils.json import DefaultEncoder
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import load


#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        JCM: callable = None,
        SvB: str = None,
        SvB_MA: str = None,
        blind: bool = False,
        apply_trigWeight: bool = True,
        apply_btagSF: bool = True,
        apply_FvT: bool = True,
        apply_boosted_veto: bool = False,
        run_lowpt_selection: bool = False,
        fill_histograms: bool = True,
        run_SvB: bool = True,
        corrections_metadata: str = "analysis/metadata/corrections.yml",
        top_reconstruction_override: bool = False,
        run_systematics: list = [],
        make_classifier_input: str = None,
        make_top_reconstruction: str = None,
        make_friend_JCM_weight: str = None,
        make_friend_FvT_weight: str = None,
        isSyntheticData: bool = False,
        subtract_ttbar_with_weights: bool = False,
        friend_trigWeight: str = None,
        friend_top_reconstruction: str = None,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.blind = blind
        self.JCM = jetCombinatoricModel(JCM) if JCM else None
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.run_lowpt_selection = run_lowpt_selection
        self.run_SvB = run_SvB
        self.fill_histograms = fill_histograms
        self.apply_boosted_veto = apply_boosted_veto
        if SvB or SvB_MA: # import torch on demand
            from analysis.helpers.networks import HCREnsemble
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        with open(corrections_metadata, "r") as f:
            self.corrections_metadata = yaml.safe_load(f)

        self.run_systematics = run_systematics
        self.make_top_reconstruction = make_top_reconstruction
        self.make_classifier_input = make_classifier_input
        self.make_friend_JCM_weight = make_friend_JCM_weight
        self.make_friend_FvT_weight = make_friend_FvT_weight
        self.top_reconstruction_override = top_reconstruction_override
        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.friend_trigWeight = friend_trigWeight
        self.friend_top_reconstruction = friend_top_reconstruction

        if self.friend_trigWeight:
            with open(friend_trigWeight, 'r') as f:
                self.friend_trigWeight = Friend.from_json(json.load(f)['trigWeight'])

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
        if self.run_SvB:
            self.cutFlowCuts += ["passSvB", "failSvB"]
            self.histCuts += ["passSvB", "failSvB"]

    def process(self, event):

        fname   = event.metadata['filename']
        self.dataset = event.metadata['dataset']
        self.estart  = event.metadata['entrystart']
        self.estop   = event.metadata['entrystop']
        self.chunk   = f'{self.dataset}::{self.estart:6d}:{self.estop:6d} >>> '
        self.year    = event.metadata['year']
        self.year_label = self.corrections_metadata[self.year]['year_label']
        self.processName = event.metadata['processName']

        if self.top_reconstruction_override:
            self.top_reconstruction = self.top_reconstruction_override
            logging.info(f"top_reconstruction overridden to {self.top_reconstruction}\n")
        else:
            self.top_reconstruction = event.metadata.get("top_reconstruction", None)

        #
        # Set process and datset dependent flags
        #
        self.config = processor_config(self.processName, self.dataset, event)
        logging.debug(f'{self.chunk} config={self.config}, for file {fname}\n')

        #
        #  If doing RW
        #
        # if self.config["isSyntheticData"] and not self.config["isPSData"]:
        #     with open(f"jet_clustering/jet-splitting-PDFs-00-08-00/hT-reweight-00-00-01/hT_weights_{self.year}.yml", "r") as f:
        #         self.hT_weights= yaml.safe_load(f)


        self.nEvent = len(event)

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split("/")[-1], "")
        if self.apply_FvT:
            if self.config["isMixedData"]:

                FvT_name = event.metadata["FvT_name"]
                event["FvT"] = getattr( NanoEventsFactory.from_root( f'{event.metadata["FvT_file"]}', entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema, ).events(),
                                        FvT_name )

                event["FvT", "FvT"] = getattr(event["FvT"], FvT_name)

                #
                # Dummies
                #
                event["FvT", "q_1234"] = np.full(len(event), -1, dtype=int)
                event["FvT", "q_1324"] = np.full(len(event), -1, dtype=int)
                event["FvT", "q_1423"] = np.full(len(event), -1, dtype=int)

            elif self.config["isDataForMixed"] or self.config["isTTForMixed"]:

                #
                # Use the first to define the FvT weights
                #
                event["FvT"] = getattr( NanoEventsFactory.from_root( f'{event.metadata["FvT_files"][0]}', entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema, ).events(),
                                        event.metadata["FvT_names"][0], )

                event["FvT", "FvT"] = getattr( event["FvT"], event.metadata["FvT_names"][0] )

                #
                # Dummies
                #
                event["FvT", "q_1234"] = np.full(len(event), -1, dtype=int)
                event["FvT", "q_1324"] = np.full(len(event), -1, dtype=int)
                event["FvT", "q_1423"] = np.full(len(event), -1, dtype=int)

                for _FvT_name, _FvT_file in zip( event.metadata["FvT_names"], event.metadata["FvT_files"] ):

                    event[_FvT_name] = getattr( NanoEventsFactory.from_root( f"{_FvT_file}", entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema, ).events(),
                                                _FvT_name, )

                    event[_FvT_name, _FvT_name] = getattr(event[_FvT_name], _FvT_name)

            else:
                event["FvT"] = ( NanoEventsFactory.from_root( f'{fname.replace("picoAOD", "FvT")}', entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema).events().FvT )

            if "std" not in event.FvT.fields:
                event["FvT", "std"] = np.ones(len(event))
                event["FvT", "pt4"] = np.ones(len(event))
                event["FvT", "pt3"] = np.ones(len(event))
                event["FvT", "pd4"] = np.ones(len(event))
                event["FvT", "pd3"] = np.ones(len(event))

            event["FvT", "frac_err"] = event["FvT"].std / event["FvT"].FvT
            if not ak.all(event.FvT.event == event.event):
                raise ValueError("ERROR: FvT events do not match events ttree")

        if self.run_SvB:
            if (self.classifier_SvB is None) | (self.classifier_SvB_MA is None):
                # SvB_file = f'{path}/SvB_newSBDef.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                SvB_file = f'{path}/SvB_ULHH.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                event["SvB"] = ( NanoEventsFactory.from_root( SvB_file,
                                                              entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema).events().SvB )

                if not ak.all(event.SvB.event == event.event):
                    raise ValueError("ERROR: SvB events do not match events ttree")

                # SvB_MA_file = f'{path}/SvB_MA_newSBDef.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                SvB_MA_file = f'{path}/SvB_MA_ULHH.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                event["SvB_MA"] = ( NanoEventsFactory.from_root( SvB_MA_file,
                                                                 entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema ).events().SvB_MA )

                if not ak.all(event.SvB_MA.event == event.event):
                    raise ValueError("ERROR: SvB_MA events do not match events ttree")

                # defining SvB for different SR
                setSvBVars("SvB", event)
                setSvBVars("SvB_MA", event)

        if self.config["isDataForMixed"]:

            #
            # Load the different JCMs
            #
            JCM_array = TreeReader( lambda x: [ s for s in x if s.startswith("pseudoTagWeight_3bDvTMix4bDvT_v") ] ).arrays(Chunk.from_coffea_events(event))

            for _JCM_load in event.metadata["JCM_loads"]:
                event[_JCM_load] = JCM_array[_JCM_load]

        #
        # Event selection
        #
        event = apply_event_selection_4b( event, self.corrections_metadata[self.year], cut_on_lumimask=self.config["cut_on_lumimask"])


        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, target=target,
                                                  do_MC_weights=self.config["do_MC_weights"],
                                                  dataset=self.dataset,
                                                  year_label=self.year_label,
                                                  estart=self.estart,
                                                  estop=self.estop,
                                                  friend_trigWeight=self.friend_trigWeight,
                                                  corrections_metadata=self.corrections_metadata[self.year],
                                                  apply_trigWeight=self.apply_trigWeight,
                                                  isTTForMixed=self.config["isTTForMixed"]
                                                 )


        #
        # Checking boosted selection (should change in the future)
        #
        event['vetoBoostedSel'] = np.full(len(event), False)
        if self.apply_boosted_veto & self.dataset.startswith("GluGluToHHTo4B_cHHH1"):
            boosted_file = load("metadata/boosted_overlap_signal.coffea")['boosted']
            boosted_events = boosted_file.get(self.dataset, {}).get('event', event.event)
            boosted_events_set = set(boosted_events)
            event['vetoBoostedSel'] = ~np.array([e in boosted_events_set for e in event.event.to_numpy()])

        if self.apply_boosted_veto and self.dataset.startswith("data"):
            boosted_file = load("metadata/boosted_overlap_data.coffea")
            boosted_runs = boosted_file.get('run', [])
            boosted_lumis = boosted_file.get('luminosityBlock', [])
            boosted_events = boosted_file.get('event', [])
            boosted_events_set = set(zip(boosted_runs, boosted_lumis, boosted_events))
            event_tuples = zip(event.run.to_numpy(), event.luminosityBlock.to_numpy(), event.event.to_numpy())
            event['vetoBoostedSel'] = ~np.array([t in boosted_events_set for t in event_tuples])
            
        #
        # Calculate and apply Jet Energy Calibration
        #
        if self.config["do_jet_calibration"]:
            jets = apply_jerc_corrections(event,
                                          corrections_metadata=self.corrections_metadata[self.year],
                                          isMC=self.config["isMC"],
                                          run_systematics=self.run_systematics,
                                          dataset=self.dataset
                                          )
        else:
            jets = event.Jet


        shifts = [({"Jet": jets}, None)]
        if self.run_systematics:
            for jesunc in self.corrections_metadata[self.year]["JES_uncertainties"]:
                shifts.extend( [ ({"Jet": jets[f"JES_{jesunc}"].up}, f"CMS_scale_j_{jesunc}Up"),
                                 ({"Jet": jets[f"JES_{jesunc}"].down}, f"CMS_scale_j_{jesunc}Down"), ] )

            shifts.extend( [({"Jet": jets.JER.up}, f"CMS_res_j_{self.year_label}Up"), ({"Jet": jets.JER.down}, f"CMS_res_j_{self.year_label}Down")] )

            logging.info(f"\nJet variations {[name for _, name in shifts]}")

        return processor.accumulate( self.process_shift(update_events(event, collections), name, weights, list_weight_names, target) for collections, name in shifts )

    def process_shift(self, event, shift_name, weights, list_weight_names, target):
        """For different jet variations. It computes event variations for the nominal case."""

        # Copy the weights to avoid modifying the original
        weights = copy.copy(weights)

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, self.corrections_metadata[self.year],
                                           doLeptonRemoval=self.config["do_lepton_jet_cleaning"],
                                           override_selected_with_flavor_bit=self.config["override_selected_with_flavor_bit"],
                                           run_lowpt_selection=self.run_lowpt_selection
                                           )


        #
        #  Test hT reweighting the synthetic data
        #
        # if self.config["isSyntheticData"] and not self.config["isPSData"]:
        #     hT_index = np.floor_divide(event.hT_selected,30).to_numpy()
        #     hT_index[hT_index > 48] = 48
        #
        #     vectorized_hT = np.vectorize(lambda i: self.hT_weights["weights"][int(i)])
        #     weights_hT = vectorized_hT(hT_index)
        #
        #     weights.add( "hT_reweight", weights_hT )
        #     list_weight_names.append(f"hT_reweight")


        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        #selections.add( "passHLT", ( np.full(len(event), True) if skip_HLT_cut else event.passHLT ) )
        selections.add( "passHLT", ( event.passHLT if self.config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( 'passJetMult', event.passJetMult )
        allcuts = [ 'lumimask', 'passNoiseFilter', 'passHLT', 'passJetMult' ]
        event['weight'] = weights.weight()   ### this is for _cutflow

        #
        #  Cut Flows
        #
        processOutput = {}
        if not shift_name:
            processOutput['nEvent'] = {}
            processOutput['nEvent'][event.metadata['dataset']] = {
                'nEvent' : self.nEvent,
                'genWeights': np.sum(event.genWeight) if self.config["isMC"] else self.nEvent

            }

            self._cutFlow = cutFlow(self.cutFlowCuts)
            self._cutFlow.fill( "all", event[selections.require(lumimask=True)], allTag=True)
            self._cutFlow.fill( "all_woTrig", event[selections.require(lumimask=True)], allTag=True,
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.require(lumimask=True)] ))
            self._cutFlow.fill( "passNoiseFilter", event[selections.require(lumimask=True, passNoiseFilter=True)], allTag=True)
            self._cutFlow.fill( "passNoiseFilter_woTrig", event[selections.require(lumimask=True, passNoiseFilter=True)], allTag=True,
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.require(lumimask=True, passNoiseFilter=True)] ))
            self._cutFlow.fill( "passHLT", event[ selections.require( lumimask=True, passNoiseFilter=True, passHLT=True ) ], allTag=True, )
            self._cutFlow.fill( "passHLT_woTrig", event[ selections.require( lumimask=True, passNoiseFilter=True, passHLT=True ) ], allTag=True,
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.require(lumimask=True, passNoiseFilter=True, passHLT=True)] ))
            self._cutFlow.fill( "passJetMult", event[ selections.all(*allcuts)], allTag=True )
            self._cutFlow.fill( "passJetMult_woTrig", event[ selections.all(*allcuts)], allTag=True,
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

        #
        # Calculate and apply btag scale factors
        #
        if self.config["isMC"] and self.apply_btagSF:

            weights, list_weight_names = add_btagweights( event, weights,
                                                         list_weight_names=list_weight_names,
                                                         shift_name=shift_name,
                                                         use_prestored_btag_SF=self.config["use_prestored_btag_SF"],
                                                         run_systematics=self.run_systematics,
                                                         corrections_metadata=self.corrections_metadata[self.year]
            )
            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )
            event["weight"] = weights.weight()
            if not shift_name:
                self._cutFlow.fill( "passJetMult_btagSF", event[selections.all(*allcuts)], allTag=True )
                self._cutFlow.fill( "passJetMult_btagSF_woTrig", event[selections.all(*allcuts)], allTag=True,
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

        #
        # Preselection: keep only three or four tag events
        #
        selections.add("passPreSel", event.passPreSel)
        allcuts.append("passPreSel")
        analysis_selections = selections.all(*allcuts)
        selev = event[analysis_selections]

        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_SvB(selev, self.dataset, self.year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*allcuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            allcuts.append("pass_ttbar_filter")
            self._cutFlow.fill( "pass_ttbar_filter", event[selections.all(*allcuts)], allTag=True )

            analysis_selections = selections.all(*allcuts)
            selev = selev[pass_ttbar_filter_selev]

        #
        #  Build the top Candiates
        #
        if self.friend_top_reconstruction:  ## temporary until we create friend trees
            with open(self.friend_top_reconstruction, 'r') as f:
                self.friend_top_reconstruction = Friend.from_json(json.load(f)[f'top_reco{"_"+shift_name if shift_name else ""}'])
            top_cand = self.friend_top_reconstruction.arrays(target)[analysis_selections]
            adding_top_reco_to_event( selev, top_cand )

        else:
            if self.top_reconstruction in ["slow","fast"]:

                # sort the jets by btagging
                selev.selJet = selev.selJet[ ak.argsort(selev.selJet.btagDeepFlavB, axis=1, ascending=False) ]

                if self.top_reconstruction == "slow":
                    top_cands = find_tops_slow(selev.selJet)
                else:
                    try:
                        top_cands = find_tops(selev.selJet)
                    except Exception as e:
                        print("WARNING: Fast top_reconstruction failed with exception: ")
                        print(f"{e}\n")
                        print("... Trying the slow top_reconstruction")
                        top_cands = find_tops_slow(selev.selJet)

                selev['top_cand'], _ = buildTop(selev.selJet, top_cands)
                ### with top friendtree we dont need the next two lines
                selev["xbW"] = selev.top_cand.xbW
                selev["xW"] = selev.top_cand.xW

        #
        #  Build di-jets and Quad-jets
        #
        create_cand_jet_dijet_quadjet( selev, event.event,
                                      isMC = self.config["isMC"],
                                      apply_FvT=self.apply_FvT,
                                      apply_boosted_veto=self.apply_boosted_veto,
                                      run_SvB=self.run_SvB,
                                      run_systematics=self.run_systematics,
                                      classifier_SvB=self.classifier_SvB,
                                      classifier_SvB_MA=self.classifier_SvB_MA,
                                      )

        #
        # Example of how to write out event numbers
        #
        # from analysis.helpers.write_debug_info import add_debug_info_to_output
        # add_debug_info_to_output(selev, processOutput)

        weights, list_weight_names = add_pseudotagweights( selev, weights,
                                                           analysis_selections,
                                                           JCM=self.JCM,
                                                           apply_FvT=self.apply_FvT,
                                                           isDataForMixed=self.config["isDataForMixed"],
                                                           list_weight_names=list_weight_names,
                                                           event_metadata=event.metadata,
                                                           year_label=self.year_label,
                                                           len_event=len(event),
                                                          )


        #
        # Blind data in fourTag SR
        #
        if not (self.config["isMC"] or "mixed" in self.dataset) and self.blind:
            blind_sel = np.full( len(event), True)
            blind_sel[ analysis_selections ] = ~(selev["quadJet_selected"].SR & selev.fourTag)
            selections.add( 'blind', blind_sel )
            allcuts.append( 'blind' )
            analysis_selections = selections.all(*allcuts)
            selev = selev[~(selev["quadJet_selected"].SR & selev.fourTag)]

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[analysis_selections]
        selev["trigWeight"] = weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
        selev["no_weight"] = np.ones(len(selev))
        if not shift_name:
            self._cutFlow.fill("passPreSel", selev)
            self._cutFlow.fill("passPreSel_woTrig", selev,
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections] ))
            self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
            self._cutFlow.fill("passDiJetMass_woTrig", selev[selev.passDiJetMass],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections][selev.passDiJetMass] ))
            if self.run_SvB:
                selev['passSR'] = selev.passDiJetMass & selev["quadJet_selected"].SR
                self._cutFlow.fill( "SR", selev[selev.passSR] )
                self._cutFlow.fill( "SR_woTrig", selev[selev.passSR],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections][selev.passSR] ))
                selev['passSB'] = selev.passDiJetMass & selev["quadJet_selected"].SB
                self._cutFlow.fill( "SB", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)] )
                self._cutFlow.fill( "SB_woTrig", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections][selev.passSB] ))
                self._cutFlow.fill("passSvB", selev[selev.passSvB])
                self._cutFlow.fill("passSvB_woTrig", selev[selev.passSvB],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections][selev.passSvB] ))
                self._cutFlow.fill("failSvB", selev[selev.failSvB])
                self._cutFlow.fill("failSvB_woTrig", selev[selev.failSvB],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections][selev.failSvB] ))

            self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        #
        # Hists
        #
        hist = {}
        if self.fill_histograms:
            if not self.run_systematics:
                ## this can be simplified
                hist = filling_nominal_histograms(selev, self.JCM,
                                                processName=self.processName,
                                                year=self.year,
                                                isMC=self.config["isMC"],
                                                histCuts=self.histCuts,
                                                apply_FvT=self.apply_FvT,
                                                run_SvB=self.run_SvB,
                                                top_reconstruction=self.top_reconstruction,
                                                isDataForMixed=self.config['isDataForMixed'],
                                                run_lowpt_selection=self.run_lowpt_selection,
                                                event_metadata=event.metadata)
            #
            # Run systematics
            #
            else:
                hist = filling_syst_histograms(selev, weights,
                                                analysis_selections,
                                                shift_name=shift_name,
                                                processName=self.processName,
                                                year=self.year,
                                                histCuts=self.histCuts)

        friends = { 'friends': {} }
        if self.make_top_reconstruction is not None:
            from ..helpers.dump_friendtrees import dump_top_reconstruction

            friends["friends"] = ( friends["friends"]
                | dump_top_reconstruction(
                    selev,
                    self.make_top_reconstruction,
                    f"top_reco{'_'+shift_name if shift_name else ''}",
                    analysis_selections,
                )
            )

        if self.make_classifier_input is not None:
            for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                selev[k] = selev["quadJet_selected"][k]
            selev["nSelJets"] = ak.num(selev.selJet)

            from ..helpers.dump_friendtrees import dump_input_friend

            friends["friends"] = ( friends["friends"]
                | dump_input_friend(
                    selev,
                    self.make_classifier_input,
                    "HCR_input",
                    analysis_selections,
                    weight="weight" if self.config["isMC"] else "weight_noJCM_noFvT",
                    NotCanJet="notCanJet_coffea",
                )
            )
        if self.make_friend_JCM_weight is not None:
            from ..helpers.dump_friendtrees import dump_JCM_weight

            friends["friends"] = ( friends["friends"]
                | dump_JCM_weight(selev, self.make_classifier_input, "JCM_weight", analysis_selections)
            )

        if self.make_friend_FvT_weight is not None:
            from ..helpers.dump_friendtrees import dump_FvT_weight

            friends["friends"] = ( friends["friends"]
                | dump_FvT_weight(selev, self.make_classifier_input, "FvT_weight", analysis_selections)
            )

        return hist | processOutput | friends

    def postprocess(self, accumulator):
        return accumulator
