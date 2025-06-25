from __future__ import annotations

import copy
import logging
import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING

import awkward as ak
import numpy as np
import yaml
from analysis.helpers.common import apply_jerc_corrections, update_events
from analysis.helpers.cutflow import cutFlow
from analysis.helpers.event_weights import (
    add_btagweights,
    add_pseudotagweights,
    add_weights,
)
from analysis.helpers.event_selection import apply_event_selection, apply_dilep_ttbar_selection, apply_4b_selection
from analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms,
)
from analysis.helpers.FriendTreeSchema import FriendTreeSchema
from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.processor_config import processor_config
from analysis.helpers.candidates_selection import create_cand_jet_dijet_quadjet
from analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_SvB
from analysis.helpers.topCandReconstruction import (
    adding_top_reco_to_event,
    buildTop,
    find_tops,
    find_tops_slow,
)
from base_class.hist import Fill
from base_class.root import Chunk, TreeReader
from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.util import load
from memory_profiler import profile

from ..helpers.load_friend import (
    FriendTemplate,
    parse_friends,
    rename_FvT_friend,
    rename_SvB_friend,
)

if TYPE_CHECKING:
    from ..helpers.classifier.HCR import HCRModelMetadata
from analysis.helpers.truth_tools import find_genpart

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


def _init_classfier(path: str | list[HCRModelMetadata]):
    if path is None:
        return None
    if isinstance(path, str):
        from ..helpers.classifier.HCR import Legacy_HCREnsemble
        return Legacy_HCREnsemble(path)
    else:
        from ..helpers.classifier.HCR import HCREnsemble
        return HCREnsemble(path)


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        SvB: str|list[HCRModelMetadata] = None,
        SvB_MA: str|list[HCRModelMetadata] = None,
        blind: bool = False,
        apply_JCM: bool = True,
        JCM_file: str = "analysis/weights/JCM/AN_24_089_v3/jetCombinatoricModel_SB_6771c35.yml",
        apply_trigWeight: bool = True,
        apply_btagSF: bool = True,
        apply_FvT: bool = True,
        apply_boosted_veto: bool = False,
        run_dilep_ttbar_crosscheck: bool = False,
        fill_histograms: bool = True,
        hist_cuts = ['passPreSel'],
        run_SvB: bool = True,
        corrections_metadata: str = "analysis/metadata/corrections.yml",
        top_reconstruction_override: bool = False,
        run_systematics: list = [],
        make_classifier_input: str = None,
        make_top_reconstruction: str = None,
        make_friend_JCM_weight: str = None,
        make_friend_FvT_weight: str = None,
        make_friend_SvB: str = None,
        subtract_ttbar_with_weights: bool = False,
        friends: dict[str, str|FriendTemplate] = None,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.blind = blind
        self.apply_JCM = jetCombinatoricModel(JCM_file) if apply_JCM else None
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.run_SvB = run_SvB
        self.fill_histograms = fill_histograms
        self.run_dilep_ttbar_crosscheck = run_dilep_ttbar_crosscheck
        self.apply_boosted_veto = apply_boosted_veto
        self.classifier_SvB = _init_classfier(SvB)
        self.classifier_SvB_MA = _init_classfier(SvB_MA)
        with open(corrections_metadata, "r") as f:
            self.corrections_metadata = yaml.safe_load(f)

        self.run_systematics = run_systematics
        self.make_top_reconstruction = make_top_reconstruction
        self.make_classifier_input = make_classifier_input
        self.make_friend_JCM_weight = make_friend_JCM_weight
        self.make_friend_FvT_weight = make_friend_FvT_weight
        self.make_friend_SvB = make_friend_SvB
        self.top_reconstruction_override = top_reconstruction_override
        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.friends = parse_friends(friends)
        self.histCuts = hist_cuts

    def process(self, event):
        logging.debug(event.metadata)
        fname   = event.metadata['filename']
        self.dataset = event.metadata['dataset']
        self.estart  = event.metadata['entrystart']
        self.estop   = event.metadata['entrystop']
        self.chunk   = f'{self.dataset}::{self.estart:6d}:{self.estop:6d} >>> '
        self.year    = event.metadata['year']
        self.year_label = self.corrections_metadata[self.year]['year_label']
        self.processName = event.metadata['processName']

        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)

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

        #
        #  If applying Gaussian Kernal to signal
        #
        self.gaussKernalMean = None
        if self.config["isSignal"] and (self.gaussKernalMean is not None) :
            bin_edges = np.linspace(0, 1200, 100)  # 100 bins from 0 to 1200
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers
            sigma = 0.05 * self.gaussKernalMean  # Standard deviation of the Gaussian
            self.resonance_weights = np.exp(-0.5 * ((bin_centers - self.gaussKernalMean) / sigma) ** 2)  # Gaussian formula

        self.nEvent = len(event)

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split("/")[-1], "")
        if self.apply_FvT:
            if "FvT" in self.friends:
                event["FvT"] = rename_FvT_friend(target, self.friends["FvT"])
                if self.config["isDataForMixed"] or self.config["isTTForMixed"]:
                    for _FvT_name in event.metadata["FvT_names"]:
                        event[_FvT_name] = rename_FvT_friend(target, self.friends[_FvT_name])
                        event[_FvT_name, _FvT_name] = event[_FvT_name].FvT
            else:
                # TODO: remove backward compatibility in the future
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
            for k in self.friends:
                if k.startswith("SvB"):
                    try:
                        event[k] = rename_SvB_friend(target, self.friends[k])
                        setSvBVars(k, event)
                    except Exception as e:
                        event[k] = self.friends[k].arrays(target)

            if "SvB" not in self.friends and self.classifier_SvB is None:
                # SvB_file = f'{path}/SvB_newSBDef.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB")}'
                SvB_file = f'{path}/SvB_ULHH.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                event["SvB"] = ( NanoEventsFactory.from_root( SvB_file,
                                                              entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema).events().SvB )

                if not ak.all(event.SvB.event == event.event):
                    raise ValueError("ERROR: SvB events do not match events ttree")
                # defining SvB for different SR
                setSvBVars("SvB", event)

            if "SvB_MA" not in self.friends and self.classifier_SvB_MA is None:
                # SvB_MA_file = f'{path}/SvB_MA_newSBDef.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_MA")}'
                SvB_MA_file = f'{path}/SvB_MA_ULHH.root' if 'mix' in self.dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                event["SvB_MA"] = ( NanoEventsFactory.from_root( SvB_MA_file,
                                                                 entry_start=self.estart, entry_stop=self.estop, schemaclass=FriendTreeSchema ).events().SvB_MA )

                if not ak.all(event.SvB_MA.event == event.event):
                    raise ValueError("ERROR: SvB_MA events do not match events ttree")
                # defining SvB for different SR
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
        event = apply_event_selection( event,
                                        self.corrections_metadata[self.year],
                                        cut_on_lumimask=self.config["cut_on_lumimask"]
                                        )


        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights(
            event, target=target,
            do_MC_weights=self.config["do_MC_weights"],
            dataset=self.dataset,
            year_label=self.year_label,
            estart=self.estart,
            estop=self.estop,
            friend_trigWeight=self.friends.get("trigWeight"),
            corrections_metadata=self.corrections_metadata[self.year],
            apply_trigWeight=self.apply_trigWeight,
            isTTForMixed=self.config["isTTForMixed"]
        )


        #
        # Checking boosted selection (should change in the future)
        #
        event['notInBoostedSel'] = np.full(len(event), True)
        if self.apply_boosted_veto:

            if self.dataset.startswith("GluGluToHHTo4B_cHHH1"):
                boosted_file = load("metadata/boosted_overlap_signal.coffea")['boosted']
                boosted_events = boosted_file.get(self.dataset, {}).get('event', event.event)
                boosted_events_set = set(boosted_events)
                event['notInBoostedSel'] = np.array([e not in boosted_events_set for e in event.event.to_numpy()])
            elif self.dataset.startswith("data"):
                boosted_file = load("metadata/boosted_overlap_data.coffea")
                mask = np.array(boosted_file['BDTcat_index']) > 0  ### > 0 is all boosted categories, 1 is most sensitive
                filtered_runs = np.array(boosted_file['run'])[mask]
                filtered_lumis = np.array(boosted_file['luminosityBlock'])[mask]
                filtered_events = np.array(boosted_file['event'])[mask]
                boosted_events_set = set(zip(filtered_runs, filtered_lumis, filtered_events))
                event_tuples = zip(event.run.to_numpy(), event.luminosityBlock.to_numpy(), event.event.to_numpy())
                event['notInBoostedSel'] = np.array([t not in boosted_events_set for t in event_tuples])
            else:
                logging.info(f"Boosted veto not applied for dataset {self.dataset}")

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
        event = apply_4b_selection( event, self.corrections_metadata[self.year],
                                    dataset=self.dataset,
                                    doLeptonRemoval=self.config["do_lepton_jet_cleaning"],
                                    override_selected_with_flavor_bit=self.config["override_selected_with_flavor_bit"],
                                    do_jet_veto_maps=self.config["do_jet_veto_maps"],
                                    isRun3=self.config["isRun3"],
                                    isMC=self.config["isMC"], ### temporary
                                    isSyntheticData=self.config["isSyntheticData"],
                                    isSyntheticMC=self.config["isSyntheticMC"],
                                    )

        if self.run_dilep_ttbar_crosscheck:
            event['passDilepTtbar'] = apply_dilep_ttbar_selection(event, isRun3=self.config["isRun3"])
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
        allcuts = [ 'lumimask', 'passNoiseFilter', 'passHLT', ]
        allcuts += [ 'passJetMult' ]
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

            #
            # Check outliers
            #
            # Checking for outliners in weights
            if self.config["isMC"]:
                tmp_weights = weights.weight()
                mean_weights = np.mean(tmp_weights)
                std_weights = np.std(tmp_weights)
                z_scores = np.abs((tmp_weights - mean_weights) / std_weights)
                pass_outliers = z_scores < 30
                event["passCleanGenWeight"] = pass_outliers
                if np.any(~pass_outliers) and std_weights > 0:
                    logging.warning(f"Outliers in weights:{tmp_weights[~pass_outliers]}, while mean is {mean_weights} and std is {std_weights} for event {event[~pass_outliers].event} in {self.dataset}\n")
                selections.add( "passCleanGenWeight", event.passCleanGenWeight)
                allcuts += ["passCleanGenWeight"]
            else:
                event['passCleanGenWeight'] = True
                selections.add( "passCleanGenWeight", event.passCleanGenWeight)

            #
            # Get Truth m4j
            #
            if self.config["isSignal"]:

                event['bfromHorZ_all']= find_genpart(event.GenPart, [5], [23, 25])

                if "status" in event.bfromHorZ_all.fields:
                    event['bfromHorZ'] = event.bfromHorZ_all[event.bfromHorZ_all.status == 23]
                else:
                    logging.warning(f"\nStatus Missing for GenParticles in dataset {self.dataset}\n")
                    event['bfromHorZ'] = event.bfromHorZ_all

                event['GenJet', 'selectedBs'] = (np.abs(event.GenJet.partonFlavour)==5)
                event['selGenBJet'] = event.GenJet[event.GenJet.selectedBs]
                event['matchedGenBJet'] = event.bfromHorZ.nearest( event.selGenBJet, threshold=10 )
                event["matchedGenBJet"] = event.matchedGenBJet[~ak.is_none(event.matchedGenBJet, axis=1)]

                event['pass4GenBJets'] = ak.num(event.matchedGenBJet) == 4
                event["truth_v4b"] = ak.where(  event.pass4GenBJets,
                                                event.matchedGenBJet.sum(axis=1),
                                                1e-10 * event.matchedGenBJet.sum(axis=1),
                                              )

                if self.gaussKernalMean is not None:
                    v4b_index = np.floor_divide(event.truth_v4b.mass, 12).to_numpy()
                    v4b_index[v4b_index > 98] = 98

                    vectorized_v4b = np.vectorize(lambda i: self.resonance_weights[int(i)])
                    weights_resonance = vectorized_v4b(v4b_index)
                    weights.add( "resonance_reweight", weights_resonance )
                    list_weight_names.append(f"resonance_reweight")

            else:
                event['pass4GenBJets'] = True


            selections.add( "pass4GenBJets", event.pass4GenBJets)

            #
            # Do the cutflow
            #
            sel_dict = OrderedDict({
                'all'               : selections.require(lumimask=True),
                'passCleanGenWeight': selections.require(lumimask=True, passCleanGenWeight=True),
                'pass4GenBJets'     : selections.require(lumimask=True, passCleanGenWeight=True, pass4GenBJets=True),
                'passNoiseFilter'   : selections.require(lumimask=True, passCleanGenWeight=True, passNoiseFilter=True),
                'passHLT'           : selections.require(lumimask=True, passCleanGenWeight=True, passNoiseFilter=True, passHLT=True),
            })
            sel_dict['passJetMult'] = selections.all(*allcuts)

            self._cutFlow = cutFlow(do_truth_hists=self.config["isSignal"])
            for cut, sel in sel_dict.items():
                self._cutFlow.fill( cut, event[sel], allTag=True )
                self._cutFlow.fill( f"{cut}_woTrig", event[sel], allTag=True,
                                    wOverride=weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[sel])


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
                               wOverride=weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] )

        #
        # Preselection: keep only three or four tag events
        #
        selections.add("passPreSel", event.passPreSel)
        allcuts.append("passPreSel")
        analysis_selections = selections.all(*allcuts)

        if not shift_name:
            self._cutFlow.fill( "passPreSel_allTag", event[selections.all(*allcuts)], allTag=True )
            self._cutFlow.fill( "passPreSel_allTag_woTrig", event[selections.all(*allcuts)], allTag=True,
                                wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

        weights, list_weight_names = add_pseudotagweights( event, weights,
                                                           JCM=self.apply_JCM,
                                                           apply_FvT=self.apply_FvT,
                                                           isDataForMixed=self.config["isDataForMixed"],
                                                           list_weight_names=list_weight_names,
                                                           event_metadata=event.metadata,
                                                           year_label=self.year_label,
                                                           len_event=len(event),
                                                          )

        #
        # Example of how to write out event numbers
        #
        # from analysis.helpers.write_debug_info import add_debug_Run3_data_early
        # add_debug_Run3_data_early(event, processOutput)
        #from analysis.helpers.write_debug_info import add_debug_Run3_data
        #add_debug_Run3_data(event, processOutput)

        selev = event[analysis_selections]
        #selev["passFvT50" ] = selev["FvT"].FvT > 50
        #selev["passFvT100"] = selev["FvT"].FvT > 100

        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_SvB(selev, self.dataset, self.year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*allcuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            allcuts.append("pass_ttbar_filter")
            if not shift_name:
                self._cutFlow.fill( "pass_ttbar_filter", event[selections.all(*allcuts)], allTag=True )
                self._cutFlow.fill( "pass_ttbar_filter_woTrig", event[selections.all(*allcuts)], allTag=True,
                                    wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))


            analysis_selections = selections.all(*allcuts)
            selev = selev[pass_ttbar_filter_selev]

        #
        #  Build the top Candiates
        #
        if friend := self.friends.get("top_reconstruction"):
            top_cand = friend.arrays(target)[analysis_selections]
            adding_top_reco_to_event( selev, top_cand )

        else:
            if self.top_reconstruction in ["slow","fast"]:

                # sort the jets by btagging
                selev.selJet = selev.selJet[ ak.argsort(selev.selJet.btagScore, axis=1, ascending=False) ]

                if self.top_reconstruction == "slow":
                    top_cands = find_tops_slow(selev.selJet)
                else:
                    try:
                        top_cands = find_tops(selev.selJet)
                    except Exception as e:
                        logging.warning("WARNING: Fast top_reconstruction failed with exception: ")
                        logging.warning(f"{e}\n")
                        logging.warning("... Trying the slow top_reconstruction")
                        top_cands = find_tops_slow(selev.selJet)

                selev['top_cand'], _ = buildTop(selev.selJet, top_cands)
                ### with top friendtree we dont need the next two lines
                selev["xbW"] = selev.top_cand.xbW
                selev["xW"] = selev.top_cand.xW

        #
        #  Build di-jets and Quad-jets
        #
        selev = create_cand_jet_dijet_quadjet( selev,
                                               apply_FvT=self.apply_FvT,
                                               run_SvB=self.run_SvB,
                                               run_systematics=self.run_systematics,
                                               classifier_SvB=self.classifier_SvB,
                                               classifier_SvB_MA=self.classifier_SvB_MA,
                                               processOutput = processOutput,
                                               isRun3=self.config["isRun3"],
                                              )



        #
        # Example of how to write out event numbers
        #
        # from analysis.helpers.write_debug_info import add_debug_info_to_output
        # add_debug_info_to_output(event, processOutput, weights, list_weight_names, analysis_selections)


        #
        # Blind data in fourTag SR
        #
        if not (self.config["isMC"] or "mix_v" in self.dataset) and self.blind:
            # blind_flag = ~(selev["quadJet_selected"].SR & selev.fourTag)
            blind_flag = ~( selev["quadJet_selected"].SR & (selev["SvB_MA"].ps_hh > 0.5) & selev.fourTag )
            blind_sel = np.full( len(event), True)
            blind_sel[ analysis_selections ] = blind_flag
            selections.add( 'blind', blind_sel )
            allcuts.append( 'blind' )

            if not shift_name:
                self._cutFlow.fill( "blind", event[selections.all(*allcuts)], allTag=True )
                self._cutFlow.fill( "blind_woTrig", event[selections.all(*allcuts)], allTag=True,
                                    wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))

            analysis_selections = selections.all(*allcuts)
            selev = selev[blind_flag]

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[analysis_selections]
        selev["trigWeight"] = weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
        selev['weight_woTrig'] = weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[analysis_selections]
        selev["no_weight"] = np.ones(len(selev))
        if not shift_name:
            self._cutFlow.fill("passPreSel", selev)
            self._cutFlow.fill("passPreSel_woTrig", selev,
                               wOverride=selev['weight_woTrig'])
            self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
            self._cutFlow.fill("passDiJetMass_woTrig", selev[selev.passDiJetMass],
                               wOverride=selev['weight_woTrig'][selev.passDiJetMass] )
            self._cutFlow.fill("boosted_veto_passPreSel", selev[selev.notInBoostedSel])
            self._cutFlow.fill("boosted_veto_SR", selev[selev.notInBoostedSel & selev["quadJet_selected"].SR])
            selev['passSR'] = selev.passDiJetMass & selev["quadJet_selected"].SR
            self._cutFlow.fill( "SR", selev[selev.passSR] )
            self._cutFlow.fill( "SR_woTrig", selev[selev.passSR],
                            wOverride=selev['weight_woTrig'][selev.passSR])
            selev['passSB'] = selev.passDiJetMass & selev["quadJet_selected"].SB
            self._cutFlow.fill( "SB", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)] )
            self._cutFlow.fill( "SB_woTrig", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)],
                            wOverride=selev['weight_woTrig'][selev.passSB] )
            if self.run_SvB:
                self._cutFlow.fill("passSvB", selev[selev.passSvB])
                self._cutFlow.fill("passSvB_woTrig", selev[selev.passSvB],
                               wOverride=selev['weight_woTrig'][selev.passSvB] )
                self._cutFlow.fill("failSvB", selev[selev.failSvB])
                self._cutFlow.fill("failSvB_woTrig", selev[selev.failSvB],
                               wOverride=selev['weight_woTrig'][selev.failSvB] )
            if self.run_dilep_ttbar_crosscheck:
                self._cutFlow.fill("passDilepTtbar", selev[selev.passDilepTtbar], allTag=True,
                               wOverride=selev['weight_noJCM_noFvT'][selev.passDilepTtbar] )

            self._cutFlow.addOutput(processOutput, event.metadata["dataset"])



        #
        # Hists
        #
        hist = {}
        if self.fill_histograms:
            if not self.run_systematics:
                ## this can be simplified
                hist = filling_nominal_histograms(
                    selev, 
                    self.apply_JCM,
                    processName=self.processName,
                    year=self.year,
                    isMC=self.config["isMC"],
                    histCuts=self.histCuts,
                    apply_FvT=self.apply_FvT,
                    run_SvB=self.run_SvB,
                    run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                    top_reconstruction=self.top_reconstruction,
                    isDataForMixed=self.config['isDataForMixed'],
                    event_metadata=event.metadata
                    )

            #
            # Run systematics
            #
            else:
                hist = filling_syst_histograms(
                    selev, 
                    weights,
                    analysis_selections,
                    shift_name=shift_name,
                    processName=self.processName,
                    year=self.year,
                    histCuts=self.histCuts
                    )

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

            weight = "weight_noJCM_noFvT"
            if weight not in selev.fields:
                weight = "weight"
            friends["friends"] = ( friends["friends"]
                | dump_input_friend(
                    selev,
                    self.make_classifier_input,
                    "HCR_input",
                    analysis_selections,
                    weight=weight,
                    NotCanJet="notCanJet_coffea",
                )
            )
        if self.make_friend_JCM_weight is not None:
            from ..helpers.dump_friendtrees import dump_JCM_weight

            friends["friends"] = ( friends["friends"]
                | dump_JCM_weight(selev, self.make_friend_JCM_weight, "JCM_weight", analysis_selections)
            )

        if self.make_friend_FvT_weight is not None:
            from ..helpers.dump_friendtrees import dump_FvT_weight

            friends["friends"] = ( friends["friends"]
                | dump_FvT_weight(selev, self.make_friend_FvT_weight, "FvT_weight", analysis_selections)
            )

        if self.make_friend_SvB is not None:
            from ..helpers.dump_friendtrees import dump_SvB

            friends["friends"] = ( friends["friends"]
                | dump_SvB(selev, self.make_friend_SvB, "SvB", analysis_selections)
                | dump_SvB(selev, self.make_friend_SvB, "SvB_MA", analysis_selections)
            )

        return hist | processOutput | friends

    def postprocess(self, accumulator):
        return accumulator
