import gc
import logging
import warnings

import awkward as ak
import numpy as np
import yaml
from analysis.helpers.common import apply_btag_sf, init_jet_factory, update_events
from analysis.helpers.filling_histograms import (
    filling_nominal_histograms,
    filling_syst_histograms
)
from analysis.helpers.event_weights import add_weights
from analysis.helpers.cutflow import cutFlow
from analysis.helpers.FriendTreeSchema import FriendTreeSchema
from analysis.helpers.hist_templates import (
    FvTHists,
    QuadJetHists,
    SvBHists,
    TopCandHists,
    WCandHists,
)
from analysis.helpers.SvB_helpers import setSvBVars, compute_SvB
from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.selection_basic_4b import (
    apply_event_selection_4b,
    apply_object_selection_4b,
)
from analysis.helpers.topCandReconstruction import (
    buildTop,
    dumpTopCandidateTestVectors,
    find_tops,
    find_tops_slow,
)
from base_class.hist import Collection, Fill
from base_class.physics.object import Elec, Jet, LorentzVector, Muon
from base_class.root import Chunk, TreeReader
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights
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
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, "r"))
        self.top_reconstruction_override = top_reconstruction_override

        isSyntheticData = False
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

    def process(self, event):

        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        # year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        isMC    = False if "data" in processName else True

        if self.top_reconstruction_override:
            self.top_reconstruction = self.top_reconstruction_override
            logging.info(f"top_reconstruction overridden to {self.top_reconstruction}")
        else:
            self.top_reconstruction = event.metadata.get("top_reconstruction", None)

        self.isMixedData    = not (dataset.find("mix_v") == -1)
        self.isDataForMixed = not (dataset.find("data_3b_for_mixed") == -1)
        self.isTTForMixed   = not (dataset.find("TTTo") == -1) and not ( dataset.find("_for_mixed") == -1 )


        nEvent = len(event)

        logging.debug(fname)
        logging.debug(f'{chunk}Process {nEvent} Events')

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split("/")[-1], "")
        if self.apply_FvT:
            if self.isMixedData:

                FvT_name = event.metadata["FvT_name"]
                event["FvT"] = getattr( NanoEventsFactory.from_root( f'{event.metadata["FvT_file"]}', entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema, ).events(),
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
                event["FvT"] = getattr( NanoEventsFactory.from_root( f'{event.metadata["FvT_files"][0]}', entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema, ).events(),
                                        event.metadata["FvT_names"][0], )

                event["FvT", "FvT"] = getattr( event["FvT"], event.metadata["FvT_names"][0] )

                #
                # Dummies
                #
                event["FvT", "q_1234"] = np.full(len(event), -1, dtype=int)
                event["FvT", "q_1324"] = np.full(len(event), -1, dtype=int)
                event["FvT", "q_1423"] = np.full(len(event), -1, dtype=int)

                for _FvT_name, _FvT_file in zip( event.metadata["FvT_names"], event.metadata["FvT_files"] ):

                    event[_FvT_name] = getattr( NanoEventsFactory.from_root( f"{_FvT_file}", entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema, ).events(),
                                                _FvT_name, )

                    event[_FvT_name, _FvT_name] = getattr(event[_FvT_name], _FvT_name)

            else:
                event["FvT"] = ( NanoEventsFactory.from_root( f'{fname.replace("picoAOD", "FvT")}', entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().FvT )

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
                # SvB_file = f'{path}/SvB_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                SvB_file = f'{path}/SvB_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                event["SvB"] = ( NanoEventsFactory.from_root( SvB_file,
                                                              entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().SvB )

                if not ak.all(event.SvB.event == event.event):
                    raise ValueError("ERROR: SvB events do not match events ttree")

                # SvB_MA_file = f'{path}/SvB_MA_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                SvB_MA_file = f'{path}/SvB_MA_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                event["SvB_MA"] = ( NanoEventsFactory.from_root( SvB_MA_file,
                                                                 entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema ).events().SvB_MA )

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
        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year], self.isMixedData)
        #
        # Checking boosted selection (should change in the future)
        #
        if self.apply_boosted_veto:
            boosted_file = load("analysis/hists/counts_boosted.coffea")['boosted']
            boosted_events = boosted_file[dataset]['event'] if dataset in boosted_file.keys() else event.event
            event['vetoBoostedSel'] = ~np.isin( event.event.to_numpy(), boosted_events )

        #
        # Calculate and apply Jet Energy Calibration
        #
        if ( self.isSyntheticData or self.isMixedData or self.isDataForMixed or self.isTTForMixed or not isMC ):  #### AGE: data corrections are not applied. Should be changed
            jets = event.Jet

        else:
            juncWS = [ self.corrections_metadata[year]["JERC"][0].replace("STEP", istep)
                       for istep in ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"] ] #+ self.corrections_metadata[year]["JERC"][2:]

            if self.run_systematics:
                juncWS += [self.corrections_metadata[year]["JERC"][1]]
            jets = init_jet_factory(juncWS, event, isMC)

        shifts = [({"Jet": jets}, None)]
        if self.run_systematics:
            for jesunc in self.corrections_metadata[year]["JES_uncertainties"]:
                shifts.extend( [ ({"Jet": jets[f"JES_{jesunc}"].up}, f"CMS_scale_j_{jesunc}Up"),
                                 ({"Jet": jets[f"JES_{jesunc}"].down}, f"CMS_scale_j_{jesunc}Down"), ] )

            # shifts.extend( [({"Jet": jets.JER.up}, f"CMS_res_j_{year_label}Up"), ({"Jet": jets.JER.down}, f"CMS_res_j_{year_label}Down")] )

            logging.info(f"\nJet variations {[name for _, name in shifts]}")

        return processor.accumulate( self.process_shift(update_events(event, collections), name) for collections, name in shifts )

    def process_shift(self, event, shift_name):
        """For different jet variations. It computes event variations for the nominal case."""

        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year        = event.metadata['year']
        year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        isMC        = True if event.run[0] == 1 else False

        nEvent = len(event)

        ### adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, isMC, dataset, year_label,
                                                 estart, estop, 
                                                 self.corrections_metadata[year],
                                                 self.apply_trigWeight,
                                                 self.isTTForMixed
                                                  ) 

        # Apply object selection (function does not remove events, adds content to objects)
        doLeptonRemoval = not (self.isMixedData or self.isTTForMixed or self.isDataForMixed)
        event = apply_object_selection_4b( event, self.corrections_metadata[year],
                                           doLeptonRemoval=doLeptonRemoval, isSyntheticData=self.isSyntheticData )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        skip_HLT_cut = (isMC or self.isMixedData or self.isTTForMixed or self.isDataForMixed)
        selections.add( "passHLT", ( np.full(len(event), True) if skip_HLT_cut else event.passHLT ) )
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
                'nEvent' : nEvent,
                'genWeights': np.sum(event.genWeight) if isMC else nEvent

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
        #### AGE to add btag JES
        #
        if isMC and self.apply_btagSF:

            if (not shift_name) & self.run_systematics:
                btag_SF_weights = apply_btag_sf( event.selJet, correction_file=self.corrections_metadata[year]["btagSF"],
                                                 btag_uncertainties=self.corrections_metadata[year][ "btag_uncertainties" ], )

                weights.add_multivariation( f"CMS_btag", btag_SF_weights["btagSF_central"],
                                            self.corrections_metadata[year]["btag_uncertainties"],
                                            [ var.to_numpy() for name, var in btag_SF_weights.items() if "_up" in name ],
                                            [ var.to_numpy() for name, var in btag_SF_weights.items() if "_down" in name ], )
            else:
                weights.add( "CMS_btag",
                         apply_btag_sf( event.selJet, correction_file=self.corrections_metadata[year]["btagSF"], btag_uncertainties=None, )["btagSF_central"], )
            list_weight_names.append(f"CMS_btag")

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

        #
        #  Calculate hT
        #
        selev["hT"] = ak.sum(selev.Jet[selev.Jet.selected_loose].pt, axis=1)
        selev["hT_selected"] = ak.sum(selev.Jet[selev.Jet.selected].pt, axis=1)
        selev["hT_trigger"] = ak.sum(selev.Jet[selev.Jet.ht_selected].pt, axis=1)

        #
        # Build and select boson candidate jets with bRegCorr applied
        #
        sorted_idx = ak.argsort( selev.Jet.btagDeepFlavB * selev.Jet.selected, axis=1, ascending=False )
        canJet_idx = sorted_idx[:, 0:4]
        notCanJet_idx = sorted_idx[:, 4:]
        canJet = selev.Jet[canJet_idx]

        # apply bJES to canJets
        canJet = canJet * canJet.bRegCorr
        canJet["bRegCorr"] = selev.Jet.bRegCorr[canJet_idx]
        canJet["btagDeepFlavB"] = selev.Jet.btagDeepFlavB[canJet_idx]
        canJet["puId"] = selev.Jet.puId[canJet_idx]
        canJet["jetId"] = selev.Jet.puId[canJet_idx]
        if isMC:
            canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]

        canJet["calibration"] = selev.Jet.calibration[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
        selev["canJet"] = canJet

        #
        #  Should be a better way to do this...
        #
        selev["canJet0"] = canJet[:, 0]
        selev["canJet1"] = canJet[:, 1]
        selev["canJet2"] = canJet[:, 2]
        selev["canJet3"] = canJet[:, 3]

        selev["v4j"] = canJet.sum(axis=1)
        # selev['v4j', 'n'] = 1
        # print(selev.v4j.n)
        # selev['Jet', 'canJet'] = False
        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        notCanJet["isSelJet"] = 1 * ( (notCanJet.pt >= 40) & (np.abs(notCanJet.eta) < 2.4) )  
        selev["notCanJet_coffea"] = notCanJet
        selev["nNotCanJet"] = ak.num(selev.notCanJet_coffea)

        #
        # calculate pseudoTagWeight for threeTag events
        #
        all_weights = ['genweight', 'CMS_bbbb_resolved_ggf_triggerEffSF', f'CMS_pileup_{year_label}' ,'CMS_btag']
        logging.debug( f"noJCM_noFVT partial {weights.partial_weight(include=all_weights)[ analysis_selections ][:10]}" )
        selev["weight_noJCM_noFvT"] = weights.partial_weight( include=all_weights )[analysis_selections]

        if self.JCM:
            selev["Jet_untagged_loose"] = selev.Jet[ selev.Jet.selected & ~selev.Jet.tagged_loose ]
            nJet_pseudotagged = np.zeros(len(selev), dtype=int)
            pseudoTagWeight = np.ones(len(selev))
            pseudoTagWeight[selev.threeTag], nJet_pseudotagged[selev.threeTag] = ( self.JCM( selev[selev.threeTag]["Jet_untagged_loose"], selev.event[selev.threeTag], ) )
            selev["nJet_pseudotagged"] = nJet_pseudotagged
            selev["pseudoTagWeight"] = pseudoTagWeight

            #
            # apply pseudoTagWeight and FvT to threeTag events
            #
            weight_noFvT = np.array(selev.weight.to_numpy(), dtype=float)
            weight_noFvT[selev.threeTag] = ( selev.weight[selev.threeTag] * selev.pseudoTagWeight[selev.threeTag] )
            selev["weight_noFvT"] = weight_noFvT

            if self.apply_FvT:

                if self.isDataForMixed:

                    #
                    #  Load the weights for each mixed sample
                    #
                    for _JCM_load, _FvT_name in zip( event.metadata["JCM_loads"], event.metadata["FvT_names"] ):
                        _weight = np.array(selev.weight.to_numpy(), dtype=float)
                        _weight[selev.threeTag] = ( selev.weight[selev.threeTag]
                                                    * getattr(selev, f"{_JCM_load}")[selev.threeTag]
                                                    * getattr(getattr(selev, _FvT_name), _FvT_name)[ selev.threeTag ] )
                        selev[f"weight_{_FvT_name}"] = _weight

                    #
                    # Use the first as the nominal weight
                    #
                    selev["weight"] = selev[f'weight_{event.metadata["FvT_names"][0]}']

                else:
                    weight = ( pseudoTagWeight[selev.threeTag] * selev.FvT.FvT[selev.threeTag] )
                    tmp_weight = np.full(len(event), 1.0)
                    tmp_weight[analysis_selections & event.threeTag] = weight
                    weights.add("FvT", tmp_weight)
                    list_weight_names.append(f"FvT")
                    logging.debug( f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n" )

            else:
                tmp_weight = np.full(len(event), 1.0)
                tmp_weight[analysis_selections] = weight_noFvT
                weights.add("no_FvT", tmp_weight)
                list_weight_names.append(f"no_FvT")
                logging.debug( f"no_FvT {weights.partial_weight(include=['no_FvT'])[:10]}\n" )

        #
        # Build diJets, indexed by diJet[event,pairing,0/1]
        #
        canJet = selev["canJet"]
        pairing = [([0, 2], [0, 1], [0, 1]), ([1, 3], [2, 3], [3, 2])]
        diJet = canJet[:, pairing[0]] + canJet[:, pairing[1]]
        diJet["st"] = canJet[:, pairing[0]].pt + canJet[:, pairing[1]].pt
        diJet["dr"] = canJet[:, pairing[0]].delta_r(canJet[:, pairing[1]])
        diJet["dphi"] = canJet[:, pairing[0]].delta_phi(canJet[:, pairing[1]])
        diJet["lead"] = canJet[:, pairing[0]]
        diJet["subl"] = canJet[:, pairing[1]]
        # Sort diJets within views to be lead st, subl st
        diJet = diJet[ak.argsort(diJet.st, axis=2, ascending=False)]
        diJetDr = diJet[ak.argsort(diJet.dr, axis=2, ascending=True)]
        # Now indexed by diJet[event,pairing,lead/subl st]

        # Compute diJetMass cut with independent min/max for lead/subl
        minDiJetMass = np.array([[[52, 50]]])
        maxDiJetMass = np.array([[[180, 173]]])
        diJet["passDiJetMass"] = (minDiJetMass < diJet.mass) & ( diJet.mass < maxDiJetMass )

        # Compute MDRs
        min_m4j_scale = np.array([[360, 235]])
        min_dr_offset = np.array([[-0.5, 0.0]])
        max_m4j_scale = np.array([[650, 650]])
        max_dr_offset = np.array([[0.5, 0.7]])
        max_dr = np.array([[1.5, 1.5]])
        m4j = np.repeat(np.reshape(np.array(selev["v4j"].mass), (-1, 1, 1)), 2, axis=2)
        diJet["passMDR"] = (min_m4j_scale / m4j + min_dr_offset < diJet.dr) & ( diJet.dr < np.maximum(max_m4j_scale / m4j + max_dr_offset, max_dr) )

        #
        # Compute consistency of diJet masses with boson masses
        #
        mZ = 91.0
        mH = 125.0
        st_bias = np.array([[[1.02, 0.98]]])
        cZ = mZ * st_bias
        cH = mH * st_bias

        diJet["xZ"] = (diJet.mass - cZ) / (0.1 * diJet.mass)
        diJet["xH"] = (diJet.mass - cH) / (0.1 * diJet.mass)

        #
        # Build quadJets
        #
        seeds = np.array(event.event)[[0, -1]].view(np.ulonglong)
        randomstate = np.random.Generator(np.random.PCG64(seeds))
        quadJet = ak.zip( { "lead": diJet[:, :, 0],
                            "subl": diJet[:, :, 1],
                            "close": diJetDr[:, :, 0],
                            "other": diJetDr[:, :, 1],
                            "passDiJetMass": ak.all(diJet.passDiJetMass, axis=2),
                            "random": randomstate.uniform(
                                low=0.1, high=0.9, size=(diJet.__len__(), 3)
                            ), } )

        quadJet["dr"] = quadJet["lead"].delta_r(quadJet["subl"])
        quadJet["dphi"] = quadJet["lead"].delta_phi(quadJet["subl"])
        quadJet["deta"] = quadJet["lead"].eta - quadJet["subl"].eta

        #
        #  Build the top Candiates
        #
        # dumpTopCandidateTestVectors(selev, logging, chunk, 15)

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

        if self.apply_FvT:
            quadJet["FvT_q_score"] = np.concatenate( ( np.reshape(np.array(selev.FvT.q_1234), (-1, 1)),
                                                       np.reshape(np.array(selev.FvT.q_1324), (-1, 1)),
                                                       np.reshape(np.array(selev.FvT.q_1423), (-1, 1)), ),
                                                     axis=1, )

        if self.run_SvB:

            if (self.classifier_SvB is not None) | (self.classifier_SvB_MA is not None):
                compute_SvB(selev, self.classifier_SvB, self.classifier_SvB_MA)

            quadJet["SvB_q_score"] = np.concatenate( ( np.reshape(np.array(selev.SvB.q_1234), (-1, 1)),
                                                       np.reshape(np.array(selev.SvB.q_1324), (-1, 1)),
                                                       np.reshape(np.array(selev.SvB.q_1423), (-1, 1)), ),
                                                     axis=1, )

            quadJet["SvB_MA_q_score"] = np.concatenate( ( np.reshape(np.array(selev.SvB_MA.q_1234), (-1, 1)),
                                                          np.reshape(np.array(selev.SvB_MA.q_1324), (-1, 1)),
                                                          np.reshape(np.array(selev.SvB_MA.q_1423), (-1, 1)), ), axis=1, )

        #
        # Compute Signal Regions
        #
        quadJet["xZZ"] = np.sqrt(quadJet.lead.xZ**2 + quadJet.subl.xZ**2)
        quadJet["xHH"] = np.sqrt(quadJet.lead.xH**2 + quadJet.subl.xH**2)
        quadJet["xZH"] = np.sqrt( np.minimum( quadJet.lead.xH**2 + quadJet.subl.xZ**2, quadJet.lead.xZ**2 + quadJet.subl.xH**2, ) )

        max_xZZ = 2.6
        max_xZH = 1.9
        max_xHH = 1.9
        quadJet["ZZSR"] = quadJet.xZZ < max_xZZ
        quadJet["ZHSR"] = quadJet.xZH < max_xZH
        quadJet["HHSR"] = ((quadJet.xHH < max_xHH) & selev.vetoBoostedSel ) if self.apply_boosted_veto else (quadJet.xHH < max_xHH)
        quadJet["SR"] = quadJet.ZZSR | quadJet.ZHSR | quadJet.HHSR
        quadJet["SB"] = quadJet.passDiJetMass & ~quadJet.SR

        #
        #  Build the close dR and other quadjets
        #    (There is Probably a better way to do this ...
        #
        arg_min_close_dr = np.argmin(quadJet.close.dr, axis=1)
        arg_min_close_dr = arg_min_close_dr.to_numpy()
        selev["quadJet_min_dr"] = quadJet[ np.array(range(len(quadJet))), arg_min_close_dr ]

        #
        # pick quadJet at random giving preference to ones which passDiJetMass and MDRs
        #
        quadJet["rank"] = ( 10 * quadJet.passDiJetMass + quadJet.lead.passMDR + quadJet.subl.passMDR + quadJet.random )
        quadJet["selected"] = quadJet.rank == np.max(quadJet.rank, axis=1)

        selev["diJet"] = diJet
        selev["quadJet"] = quadJet
        selev["quadJet_selected"] = quadJet[quadJet.selected][:, 0]
        selev["passDiJetMass"] = ak.any(quadJet.passDiJetMass, axis=1)

        selev["region"] = ( selev["quadJet_selected"].SR * 0b10 + selev["quadJet_selected"].SB * 0b01 )

        #
        # Example of how to write out event numbers
        #
        #  passSR = (selev["quadJet_selected"].SR)
        #  passSR = (selev["SR"])
        #
        # out_data = {}
        # out_data["SvB"    ] = selev["SvB_MA"].ps[passSR]
        # out_data["event"  ] = selev["event"][passSR]
        # out_data["run"    ] = selev["run"][passSR]
        #
        # debug_mask = ~event.passJetMult
        # debug_mask = ((event["event"] == 66688  ) |
        #               (event["event"] == 249987 ) |
        #               (event["event"] == 121603 ) |
        #               (event["event"] == 7816   ) |
        #               (event["event"] == 25353  ) |
        #               (event["event"] == 165389 ) |
        #               (event["event"] == 293138 ) |
        #               (event["event"] == 150164 ) |
        #               (event["event"] == 262806 ) |
        #               (event["event"] == 281111 ) )
        #
        # out_data["debug_event"  ] = event["event"][debug_mask]
        # out_data["debug_run"    ] = event["run"][debug_mask]
        # out_data["debug_jet_pt"    ] = event.Jet[event.Jet.selected_eta].pt[debug_mask].to_list()
        # out_data["debug_jet_eta"   ] = event.Jet[event.Jet.selected_eta].eta[debug_mask].to_list()
        # out_data["debug_jet_phi"   ] = event.Jet[event.Jet.selected_eta].phi[debug_mask].to_list()
        # out_data["debug_jet_pu"    ] = event.Jet[event.Jet.selected_eta].pileup[debug_mask].to_list()
        # out_data["debug_jet_jetId" ] = event.Jet[event.Jet.selected_eta].jetId[debug_mask].to_list()
        # out_data["debug_jet_lep"   ] = event.Jet[event.Jet.selected_eta].lepton_cleaned[debug_mask].to_list()
        #
        # for out_k, out_v in out_data.items():
        #     processOutput[out_k] = {}
        #     processOutput[out_k][event.metadata['dataset']] = list(out_v)

        if self.run_SvB:
            selev["passSvB"] = selev["SvB_MA"].ps > 0.80
            selev["failSvB"] = selev["SvB_MA"].ps < 0.05

        #
        # Blind data in fourTag SR
        #
        if not (isMC or "mixed" in dataset) and self.blind:
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

            hist = filling_nominal_histograms(selev, processName, year, isMC, self.histCuts, self.apply_FvT, self.JCM, self.run_SvB, self.top_reconstruction, self.isMixedData, self.isDataForMixed, self.isTTForMixed, event.metadata)

            garbage = gc.collect()
            # print('Garbage:',garbage)

            friends = {}
            if self.make_classifier_input is not None:
                _all_selection = analysis_selections
                for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                    selev[k] = selev["quadJet_selected"][k]
                selev["nSelJets"] = ak.num(selev.selJet)

                ####
                from ..helpers.classifier.HCR import dump_input_friend, dump_JCM_weight, dump_FvT_weight

                friends["friends"] = dump_input_friend( selev, self.make_classifier_input, "HCR_input", _all_selection, weight="weight" if isMC else "weight_noJCM_noFvT", NotCanJet="notCanJet_coffea") | dump_JCM_weight( selev, self.make_classifier_input, "JCM_weight", _all_selection) | dump_FvT_weight( selev, self.make_classifier_input, "FvT_weight", _all_selection)

            output = hist | processOutput | friends
        #
        # Run systematics
        #
        else:
            hist_SvB = filling_syst_histograms(shift_name, selev, processName, year, weights, analysis_selections, self.histCuts)
            garbage = gc.collect()
            # print('Garbage:',garbage)
            output = hist_SvB | processOutput

        return output

    def postprocess(self, accumulator):
        return accumulator
