import logging
import warnings

import awkward as ak
import numpy as np
import yaml, json
from analysis.helpers.common import init_jet_factory, update_events
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
        JCM=None,
        SvB=None,
        SvB_MA=None,
        blind=False,
        apply_trigWeight=True,
        apply_btagSF=True,
        apply_FvT=True,
        apply_boosted_veto=False,
        run_SvB=True,
        corrections_metadata="analysis/metadata/corrections.yml",
        top_reconstruction_override = False,
        run_systematics=[],
        make_classifier_input: str = None,
        isSyntheticData=False,
        subtract_ttbar_with_weights = False,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.blind = blind
        self.JCM = jetCombinatoricModel(JCM) if JCM else None
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.run_SvB = run_SvB
        self.apply_boosted_veto = apply_boosted_veto
        if SvB or SvB_MA: # import torch on demand
            from analysis.helpers.networks import HCREnsemble
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        with open(corrections_metadata, "r") as f:
            self.corrections_metadata = yaml.safe_load(f)
        self.top_reconstruction_override = top_reconstruction_override
        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights

        self.isSyntheticData = isSyntheticData

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

        self.run_systematics = run_systematics
        self.make_classifier_input = make_classifier_input

        # with open("hists/local/friends.json", 'r') as f:
        #     self.friend = Friend.from_json(json.load(f)['trigWeight'])

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
        # Set process type flags
        #
        self.isMC     = False if "data"    in self.processName else True
        self.isPSData = True  if "ps_data" in self.processName else False
        self.isMixedData    = not (self.dataset.find("mix_v") == -1)
        if self.isMixedData:
            self.isMC = False
        self.isDataForMixed = not (self.dataset.find("data_3b_for_mixed") == -1)
        self.isTTForMixed   = not (self.dataset.find("TTTo") == -1) and not ( self.dataset.find("_for_mixed") == -1 )


        #
        #  Nominal config (...what we would do for data)
        #
        self.cut_on_lumimask         = True
        self.cut_on_HLT_decision     = True
        self.do_MC_weights           = False
        self.do_jet_calibration      = False
        self.do_lepton_jet_cleaning  = True
        self.override_selected_with_flavor_bit  = False
        self.use_prestored_btag_SF  = False

        if self.isMC:
            self.cut_on_lumimask     = False
            self.cut_on_HLT_decision = False
            self.do_jet_calibration  = True
            self.do_MC_weights       = True


        if self.isSyntheticData:
            self.do_lepton_jet_cleaning  = False
            self.override_selected_with_flavor_bit  = True

            if self.isMC:
                self.do_jet_calibration     = False
                self.do_MC_weights          = True
                self.use_prestored_btag_SF  = True
                self.cut_on_HLT_decision    = False


        if self.isPSData:
            self.cut_on_lumimask     = False
            self.cut_on_HLT_decision = False


        if self.isMixedData:
            self.cut_on_lumimask     = False
            self.cut_on_HLT_decision = False
            self.do_lepton_jet_cleaning  = False

        if self.isTTForMixed:
            self.cut_on_lumimask        = False
            self.cut_on_HLT_decision    = False
            self.do_lepton_jet_cleaning = False
            self.do_jet_calibration     = False

        if self.isDataForMixed:
            self.cut_on_HLT_decision = False
            self.do_lepton_jet_cleaning  = False


        logging.debug(f'{self.chunk} isData={False}, isMC={self.isMC}, isMixedData={self.isMixedData}, isDataForMixed={self.isDataForMixed}, isTTForMixed={self.isTTForMixed},  isSyntheticData={self.isSyntheticData}, isPSData={self.isPSData} for file {fname}\n')
        logging.debug(f'{self.chunk} isMC {self.isMC}, isSyntheticData {self.isSyntheticData}, isPSData={self.isPSData}\n\n')




        self.nEvent = len(event)

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split("/")[-1], "")
        if self.apply_FvT:
            if self.isMixedData:

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

            elif self.isDataForMixed or self.isTTForMixed:

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

        if self.isDataForMixed:

            #
            # Load the different JCMs
            #
            JCM_array = TreeReader( lambda x: [ s for s in x if s.startswith("pseudoTagWeight_3bDvTMix4bDvT_v") ] ).arrays(Chunk.from_coffea_events(event))

            for _JCM_load in event.metadata["JCM_loads"]:
                event[_JCM_load] = JCM_array[_JCM_load]

        #
        # Event selection
        #
        event = apply_event_selection_4b( event, self.corrections_metadata[self.year], cut_on_lumimask=self.cut_on_lumimask)


        # target = Chunk.from_coffea_events(event)
        # event['tmptrigWeight'] = self.friend.arrays(target)

        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, self.do_MC_weights, self.dataset, self.year_label,
                                                  self.estart, self.estop,
                                                  self.corrections_metadata[self.year],
                                                  self.apply_trigWeight,
                                                  self.isTTForMixed
                                                 )
        #
        # Checking boosted selection (should change in the future)
        #
        if self.apply_boosted_veto:
            boosted_file = load("analysis/hists/counts_boosted.coffea")['boosted']
            boosted_events = boosted_file[self.dataset]['event'] if self.dataset in boosted_file.keys() else event.event
            event['vetoBoostedSel'] = ~np.isin( event.event.to_numpy(), boosted_events )

        #
        # Calculate and apply Jet Energy Calibration
        #
        if self.do_jet_calibration:
            juncWS = [ self.corrections_metadata[self.year]["JERC"][0].replace("STEP", istep)
                       for istep in ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"] ] #+ self.corrections_metadata[self.year]["JERC"][2:]

            if self.run_systematics:
                juncWS += [self.corrections_metadata[self.year]["JERC"][1]]
            jets = init_jet_factory(juncWS, event, self.isMC)
        else:
            jets = event.Jet

        shifts = [({"Jet": jets}, None)]
        if self.run_systematics:
            for jesunc in self.corrections_metadata[self.year]["JES_uncertainties"]:
                shifts.extend( [ ({"Jet": jets[f"JES_{jesunc}"].up}, f"CMS_scale_j_{jesunc}Up"),
                                 ({"Jet": jets[f"JES_{jesunc}"].down}, f"CMS_scale_j_{jesunc}Down"), ] )

            # shifts.extend( [({"Jet": jets.JER.up}, f"CMS_res_j_{self.year_label}Up"), ({"Jet": jets.JER.down}, f"CMS_res_j_{self.year_label}Down")] )

            logging.info(f"\nJet variations {[name for _, name in shifts]}")

        return processor.accumulate( self.process_shift(update_events(event, collections), name, weights, list_weight_names) for collections, name in shifts )

    def process_shift(self, event, shift_name, weights, list_weight_names):
        """For different jet variations. It computes event variations for the nominal case."""


        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, self.corrections_metadata[self.year],
                                           doLeptonRemoval=self.do_lepton_jet_cleaning, override_selected_with_flavor_bit=self.override_selected_with_flavor_bit )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        #selections.add( "passHLT", ( np.full(len(event), True) if skip_HLT_cut else event.passHLT ) )
        selections.add( "passHLT", ( event.passHLT if self.cut_on_HLT_decision else np.full(len(event), True)  ) )
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
                'genWeights': np.sum(event.genWeight) if self.isMC else self.nEvent

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
        if self.isMC and self.apply_btagSF:

            weights, list_weight_names = add_btagweights( event, weights,
                                                         list_weight_names=list_weight_names,
                                                         shift_name=shift_name,
                                                         use_prestored_btag_SF=self.use_prestored_btag_SF,
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
        if self.top_reconstruction in ["slow","fast"]:

            # sort the jets by btagging
            selev.selJet = selev.selJet[ ak.argsort(selev.selJet.btagDeepFlavB, axis=1, ascending=False) ]

            if self.top_reconstruction == "slow":
                top_cands = find_tops_slow(selev.selJet)
            else:
                top_cands = find_tops(selev.selJet)

            selev['top_cand'], _ = buildTop(selev.selJet, top_cands)

            selev["xbW"] = selev.top_cand.xbW
            selev["xW"] = selev.top_cand.xW

        #
        #  Build di-jets and Quad-jets
        #
        create_cand_jet_dijet_quadjet( selev, event.event,
                                      isMC = self.isMC,
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


        if self.JCM:
            weights, list_weight_names = add_pseudotagweights( selev, weights,
                                                            analysis_selections,
                                                            JCM=self.JCM,
                                                            apply_FvT=self.apply_FvT,
                                                            isDataForMixed=self.isDataForMixed,
                                                            list_weight_names=list_weight_names,
                                                            event_metadata=event.metadata,
                                                            year_label=self.year_label,
                                                            len_event=len(event),
            )

        #
        # Blind data in fourTag SR
        #
        if not (self.isMC or "mixed" in self.dataset) and self.blind:
            blind_sel = np.full( len(event), True)
            blind_sel[ analysis_selections ] = ~(selev["quadJet_selected"].SR & selev.fourTag)
            selections.add( 'blind', blind_sel )
            allcuts.append( 'blind' )
            selev = selev[~(selev["quadJet_selected"].SR & selev.fourTag)]

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[analysis_selections]
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

        if not self.run_systematics:
            ## this can be simplified
            hist = filling_nominal_histograms(selev, self.JCM,
                                              processName=self.processName,
                                              year=self.year,
                                              isMC=self.isMC,
                                              histCuts=self.histCuts,
                                              apply_FvT=self.apply_FvT,
                                              run_SvB=self.run_SvB,
                                              top_reconstruction=self.top_reconstruction,
                                              isDataForMixed=self.isDataForMixed,
                                              event_metadata=event.metadata)


            friends = {}
            if self.make_classifier_input is not None:
                _all_selection = analysis_selections
                for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                    selev[k] = selev["quadJet_selected"][k]
                selev["nSelJets"] = ak.num(selev.selJet)

                ####
                from ..helpers.dump_friendtrees import dump_input_friend, dump_JCM_weight, dump_FvT_weight

                friends["friends"] = dump_input_friend( selev, self.make_classifier_input, "HCR_input", _all_selection, weight="weight" if self.isMC else "weight_noJCM_noFvT", NotCanJet="notCanJet_coffea") | dump_JCM_weight( selev, self.make_classifier_input, "JCM_weight", _all_selection) | dump_FvT_weight( selev, self.make_classifier_input, "FvT_weight", _all_selection)

            output = hist | processOutput | friends

        #
        # Run systematics
        #
        else:
            hist_SvB = filling_syst_histograms(selev, weights,
                                               analysis_selections,
                                               shift_name=shift_name,
                                               processName=self.processName,
                                               year=self.year,
                                               histCuts=self.histCuts)
            output = hist_SvB | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
