import time
import gc
import awkward as ak
import numpy as np
import correctionlib
import yaml
import warnings
import uproot

from analysis.helpers.networks import HCREnsemble
from analysis.helpers.topCandReconstruction import find_tops, dumpTopCandidateTestVectors, buildTop, mW, mt, find_tops_slow
from analysis.helpers.clustering import cluster_bs, make_synthetic_event
    
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
from coffea.util import load
from coffea.analysis_tools import Weights, PackedSelection

from base_class.hist import Collection, Fill
from base_class.physics.object import LorentzVector, Jet, Muon, Elec
from analysis.helpers.hist_templates import SvBHists, FvTHists, QuadJetHists, WCandHists, TopCandHists


from analysis.helpers.cutflow import cutFlow
from analysis.helpers.FriendTreeSchema import FriendTreeSchema

from analysis.helpers.jetCombinatoricModel import jetCombinatoricModel
from analysis.helpers.common import init_jet_factory, apply_btag_sf, update_events

from analysis.helpers.selection_basic_4b import (
    apply_event_selection_4b,
    apply_object_selection_4b
)

import logging

from base_class.root import TreeReader, Chunk

#
# Setup
#
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


def setSvBVars(SvBName, event):

    event[SvBName, "passMinPs"] = ( (getattr(event, SvBName).pzz > 0.01)
                                    | (getattr(event, SvBName).pzh > 0.01)
                                    | (getattr(event, SvBName).phh > 0.01) )

    event[SvBName, "zz"] = ( getattr(event, SvBName).pzz >  getattr(event, SvBName).pzh ) & (getattr(event, SvBName).pzz > getattr(event, SvBName).phh)

    event[SvBName, "zh"] = ( getattr(event, SvBName).pzh >  getattr(event, SvBName).pzz ) & (getattr(event, SvBName).pzh > getattr(event, SvBName).phh)

    event[SvBName, "hh"] = ( getattr(event, SvBName).phh >= getattr(event, SvBName).pzz ) & (getattr(event, SvBName).phh >= getattr(event, SvBName).pzh)


    #
    #  Set ps_{bb}
    #
    this_ps_zz = np.full(len(event), -1, dtype=float)
    this_ps_zz[getattr(event, SvBName).zz] = getattr(event, SvBName).ps[ getattr(event, SvBName).zz ]

    this_ps_zz[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_zz"] = this_ps_zz

    this_ps_zh = np.full(len(event), -1, dtype=float)
    this_ps_zh[getattr(event, SvBName).zh] = getattr(event, SvBName).ps[ getattr(event, SvBName).zh ]

    this_ps_zh[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_zh"] = this_ps_zh

    this_ps_hh = np.full(len(event), -1, dtype=float)
    this_ps_hh[getattr(event, SvBName).hh] = getattr(event, SvBName).ps[ getattr(event, SvBName).hh ]

    this_ps_hh[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_hh"] = this_ps_hh



class analysis(processor.ProcessorABC):
    def __init__(
            self,
            *,
            JCM=None,
            SvB=None,
            SvB_MA=None,
            threeTag=True,
            apply_trigWeight=True,
            apply_btagSF=True,
            apply_FvT=True,
            apply_boosted_veto=False,
            run_SvB=True,
            do_gbb_only=True,
            do_declustering=False,
            corrections_metadata="analysis/metadata/corrections.yml",
            run_systematics=[],
            make_classifier_input: str = None,
    ):

        logging.debug("\nInitialize Analysis Processor")
        self.blind = False
        self.JCM = jetCombinatoricModel(JCM) if JCM else None
        self.apply_trigWeight = apply_trigWeight
        self.apply_btagSF = apply_btagSF
        self.apply_FvT = apply_FvT
        self.run_SvB = run_SvB
        self.apply_boosted_veto = apply_boosted_veto
        self.classifier_SvB = HCREnsemble(SvB) if SvB else None
        self.classifier_SvB_MA = HCREnsemble(SvB_MA) if SvB_MA else None
        self.corrections_metadata = yaml.safe_load(open(corrections_metadata, "r"))
        self.do_declustering = do_declustering
        self.do_gbb_only = do_gbb_only
        
        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passJetMult_btagSF",
            "passPreSel",
            "passFourTag",
            "pass0OthJets",
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

        tstart = time.time()
        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year    = event.metadata['year']
        year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        isMC    = True if event.run[0] == 1 else False

        self.top_reconstruction = event.metadata.get("top_reconstruction", None)

        isMixedData    = not (dataset.find("mix_v") == -1)
        isDataForMixed = not (dataset.find("data_3b_for_mixed") == -1)
        isTTForMixed   = not (dataset.find("TTTo") == -1) and not ( dataset.find("_for_mixed") == -1 )

        nEvent = len(event)

        logging.debug(fname)
        logging.debug(f'{chunk}Process {nEvent} Events')

        #
        # Reading SvB friend trees
        #
        path = fname.replace(fname.split("/")[-1], "")
        if self.apply_FvT:
            if isMixedData:

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

            elif isDataForMixed or isTTForMixed:

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
                #SvB_file = f'{path}/SvB_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                SvB_file = f'{path}/SvB_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_ULHH")}'
                event["SvB"] = ( NanoEventsFactory.from_root( SvB_file,
                                                              entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema).events().SvB )

                if not ak.all(event.SvB.event == event.event):
                    raise ValueError("ERROR: SvB events do not match events ttree")

                #SvB_MA_file = f'{path}/SvB_MA_newSBDef.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                SvB_MA_file = f'{path}/SvB_MA_ULHH.root' if 'mix' in dataset else f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
                event["SvB_MA"] = ( NanoEventsFactory.from_root( SvB_MA_file,
                                                                 entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema ).events().SvB_MA )

                if not ak.all(event.SvB_MA.event == event.event):
                    raise ValueError("ERROR: SvB_MA events do not match events ttree")

                # defining SvB for different SR
                setSvBVars("SvB", event)
                setSvBVars("SvB_MA", event)

        if isDataForMixed:

            #
            # Load the different JCMs
            #
            JCM_array = TreeReader( lambda x: [ s for s in x if s.startswith("pseudoTagWeight_3bDvTMix4bDvT_v") ] ).arrays(Chunk.from_coffea_events(event))

            for _JCM_load in event.metadata["JCM_loads"]:
                event[_JCM_load] = JCM_array[_JCM_load]

        #
        # Event selection
        #
        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year], isMixedData)

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
        if ( isMixedData or isDataForMixed or isTTForMixed or not isMC ):  #### AGE: data corrections are not applied. Should be changed
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

        fname   = event.metadata['filename']
        dataset = event.metadata['dataset']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        year        = event.metadata['year']
        year_label = self.corrections_metadata[year]['year_label']
        processName = event.metadata['processName']
        isMC        = True if event.run[0] == 1 else False
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)

        isMixedData    = not (dataset.find("mix_v") == -1)
        isDataForMixed = not (dataset.find("data_3b_for_mixed") == -1)
        isTTForMixed   = not (dataset.find("TTTo") == -1) and not ( dataset.find("_for_mixed") == -1 )
        nEvent = len(event)
        weights = Weights(len(event), storeIndividual=True)
        list_weight_names = []

        #
        # general event weights
        #
        if isMC:
            # genWeight
            genEventSumw = event.metadata["genEventSumw"]
            weights.add( "genweight", event.genWeight * (lumi * xs * kFactor / genEventSumw) )
            list_weight_names.append('genweight')
            logging.debug( f"genweight {weights.partial_weight(include=['genweight'])[:10]}\n" )

            # trigger Weight (to be updated)
            if self.apply_trigWeight:
                if "GluGlu" in dataset:
                    ### this is temporary until trigWeight is computed in new code
                    trigWeight_file = uproot.open(f'{fname.replace("picoAOD", "trigWeights")}')['Events']
                    trigWeight = trigWeight_file.arrays(['event', 'trigWeight_Data', 'trigWeight_MC'], entry_start=estart,entry_stop=estop)

                    if not ak.all(trigWeight.event == event.event):
                        raise ValueError('trigWeight events do not match events ttree')

                    weights.add( 'CMS_bbbb_resolved_ggf_triggerEffSF', trigWeight["trigWeight_Data"], trigWeight["trigWeight_MC"], ak.where(event.passHLT, 1., 0.) )

                else:
                    weights.add( "CMS_bbbb_resolved_ggf_triggerEffSF", event.trigWeight.Data, event.trigWeight.MC, ak.where(event.passHLT, 1.0, 0.0), )
                list_weight_names.append('CMS_bbbb_resolved_ggf_triggerEffSF')
                logging.debug( f"trigWeight {weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[:10]}\n" )


            # puWeight (to be checked)
            if not isTTForMixed:
                puWeight = list( correctionlib.CorrectionSet.from_file( self.corrections_metadata[year]["PU"] ).values() )[0]
                weights.add( f"CMS_pileup_{year_label}",
                             puWeight.evaluate(event.Pileup.nTrueInt.to_numpy(), "nominal"),
                             puWeight.evaluate(event.Pileup.nTrueInt.to_numpy(), "up"),
                             puWeight.evaluate(event.Pileup.nTrueInt.to_numpy(), "down"), )
                list_weight_names.append(f"CMS_pileup_{year_label}")
                logging.debug( f"PU weight {weights.partial_weight(include=[f'CMS_pileup_{year_label}'])[:10]}\n" )


            # L1 prefiring weight
            if ( "L1PreFiringWeight" in event.fields ):  #### AGE: this should be temprorary (field exists in UL)
                weights.add( f"CMS_prefire_{year_label}",
                             event.L1PreFiringWeight.Nom,
                             event.L1PreFiringWeight.Up,
                             event.L1PreFiringWeight.Dn, )
                logging.debug( f"L1Prefire weight {weights.partial_weight(include=[f'CMS_prefire_{year_label}'])[:10]}\n" )
                list_weight_names.append(f"CMS_prefire_{year_label}")

            if ( "PSWeight" in event.fields ):  #### AGE: this should be temprorary (field exists in UL)
                nom      = np.ones(len(weights.weight()))
                up_isr   = np.ones(len(weights.weight()))
                down_isr = np.ones(len(weights.weight()))
                up_fsr   = np.ones(len(weights.weight()))
                down_fsr = np.ones(len(weights.weight()))

                if len(event.PSWeight[0]) == 4:
                    up_isr   = event.PSWeight[:, 0]
                    down_isr = event.PSWeight[:, 2]
                    up_fsr   = event.PSWeight[:, 1]
                    down_fsr = event.PSWeight[:, 3]

                else:
                    logging.warning( f"PS weight vector has length {len(event.PSWeight[0])}" )

                weights.add("ps_isr", nom, up_isr, down_isr)
                weights.add("ps_fsr", nom, up_fsr, down_fsr)
                list_weight_names.append(f"ps_isr")
                list_weight_names.append(f"ps_fsr")

            if "LHEPdfWeight" in event.fields:

                # https://github.com/nsmith-/boostedhiggs/blob/a33dca8464018936fbe27e86d52c700115343542/boostedhiggs/corrections.py#L53
                nom  = np.ones(len(weights.weight()))
                up   = np.ones(len(weights.weight()))
                down = np.ones(len(weights.weight()))

                # NNPDF31_nnlo_hessian_pdfas
                # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
                if "306000 - 306102" in event.LHEPdfWeight.__doc__:
                    # Hessian PDF weights
                    # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
                    arg = event.LHEPdfWeight[:, 1:-2] - np.ones( (len(weights.weight()), 100) )

                    summed = ak.sum(np.square(arg), axis=1)
                    pdf_unc = np.sqrt((1.0 / 99.0) * summed)
                    weights.add("pdf_Higgs_ggHH", nom, pdf_unc + nom)

                    # alpha_S weights
                    # Eq. 27 of same ref
                    as_unc = 0.5 * ( event.LHEPdfWeight[:, 102] - event.LHEPdfWeight[:, 101] )

                    weights.add("alpha_s", nom, as_unc + nom)

                    # PDF + alpha_S weights
                    # Eq. 28 of same ref
                    pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
                    weights.add("PDFaS", nom, pdfas_unc + nom)

                else:
                    weights.add("alpha_s", nom, up, down)
                    weights.add("pdf_Higgs_ggHH", nom, up, down)
                    weights.add("PDFaS", nom, up, down)
                list_weight_names.append(f"alpha_s")
                list_weight_names.append(f"pdf_Higgs_ggHH")
                list_weight_names.append(f"PDFaS")

        else:
            weights.add("data", np.ones(len(event)))
            list_weight_names.append(f"data")

        logging.debug(f"weights event {weights.weight()[:10]}")
        logging.debug(f"Weight Statistics {weights.weightStatistics}")


        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year],
                                           isMixedData=isMixedData, isTTForMixed=isTTForMixed, isDataForMixed=isDataForMixed )


        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if (isMC or isMixedData or isTTForMixed or isDataForMixed) else event.passHLT ) )
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
        #selections.add("passPreSel", event.passPreSel)
        selections.add("passFourTag", event.fourTag)


        event['pass0OthJets'] = event.nJet_selected == 4
        selections.add("pass0OthJets", event.pass0OthJets)
        allcuts.append("passFourTag")
        allcuts.append("pass0OthJets")

        selev = event[selections.all(*allcuts)]

        #
        #  Calculate hT
        #
        selev["hT"] = ak.sum(selev.Jet[selev.Jet.selected_loose].pt, axis=1)
        selev["hT_selected"] = ak.sum(selev.Jet[selev.Jet.selected].pt, axis=1)

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
        if not isMixedData and not isTTForMixed and not isDataForMixed:
            canJet["calibration"] = selev.Jet.calibration[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]

        
        if self.do_declustering:
            canJet["jet_flavor"] = "b"
    
            clustered_jets, _clustered_splittings = cluster_bs(canJet, debug=False)
            
            if self.do_gbb_only:
    
                #
                # 1st replace bstar splittings with their original jets (b, g_bb)
                #
                bstar_mask_splittings = _clustered_splittings.jet_flavor == "bstar"
                bs_from_bstar = _clustered_splittings[bstar_mask_splittings].part_A
                gbbs_from_bstar = _clustered_splittings[bstar_mask_splittings].part_B
                jets_from_bstar = ak.concatenate([bs_from_bstar, gbbs_from_bstar], axis=1)
    
                bstar_mask = clustered_jets.jet_flavor == "bstar"
                clustered_jets_nobStar = clustered_jets[~bstar_mask]
                clustered_jets          = ak.concatenate([clustered_jets_nobStar, jets_from_bstar], axis=1)
    
    
            #
            # Declustering
            #
    
            #
            #  Read in the pdfs
            #
            #  Make with ../.ci-workflows/synthetic-dataset-plot-job.sh
            #input_pdf_file_name = "analysis/plots_synthetic_datasets/clustering_pdfs.yml"
            input_pdf_file_name = "analysis/plots_synthetic_datasets/clustering_pdfs_vs_pT.yml"
            with open(input_pdf_file_name, 'r') as input_file:
                input_pdfs = yaml.safe_load(input_file)
    
            declustered_jets = make_synthetic_event(clustered_jets, input_pdfs)
            canJet = declustered_jets
    
            #
            #  Hack
            #
            canJet["puId"] = 7
            canJet["jetId"] = 7 # selev.Jet.puId[canJet_idx]
            canJet["btagDeepFlavB"] = 1.0 # Set bs to 1 and ls to 0 
            canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]
            
        #
        #   Proceed as normal
        #
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

        notCanJet["isSelJet"] = 1 * ( (notCanJet.pt > 40) & (np.abs(notCanJet.eta) < 2.4) )  # should have been defined as notCanJet.pt>=40, too late to fix this now...
        selev["notCanJet_coffea"] = notCanJet
        selev["nNotCanJet"] = ak.num(selev.notCanJet_coffea)

        #
        # calculate pseudoTagWeight for threeTag events
        #
        all_weights = ['genweight', 'CMS_bbbb_resolved_ggf_triggerEffSF', f'CMS_pileup_{year_label}' ,'CMS_btag']
        logging.debug( f"noJCM_noFVT partial {weights.partial_weight(include=all_weights)[ selections.all(*allcuts) ][:10]}" )
        selev["weight_noJCM_noFvT"] = weights.partial_weight( include=all_weights )[selections.all(*allcuts)]

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

                if isDataForMixed:

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
                    tmp_weight[selections.all(*allcuts) & event.threeTag] = weight
                    weights.add("FvT", tmp_weight)
                    list_weight_names.append(f"FvT")
                    logging.debug( f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n" )

            else:
                tmp_weight = np.full(len(event), 1.0)
                tmp_weight[selections.all(*allcuts)] = weight_noFvT
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

            rec_top_cands = buildTop(selev.selJet, top_cands)

            selev["top_cand"] = rec_top_cands[:, 0]
            bReg_p = selev.top_cand.b * selev.top_cand.b.bRegCorr
            selev["top_cand", "p"] = bReg_p + selev.top_cand.j + selev.top_cand.l

            # mW, mt = 80.4, 173.0
            selev["top_cand", "W"] = ak.zip( { "p": selev.top_cand.j + selev.top_cand.l,
                                               "j": selev.top_cand.j,
                                               "l": selev.top_cand.l, } )

            selev["top_cand", "W", "pW"] = selev.top_cand.W.p * ( mW / selev.top_cand.W.p.mass )
            selev["top_cand", "mbW"] = (bReg_p + selev.top_cand.W.pW).mass
            selev["top_cand", "xt"] = (selev.top_cand.p.mass - mt) / ( 0.10 * selev.top_cand.p.mass )
            selev["top_cand", "xWt"] = np.sqrt(selev.top_cand.xW**2 + selev.top_cand.xt**2)
            selev["top_cand", "mbW"] = (bReg_p + selev.top_cand.W.pW).mass
            selev["top_cand", "xWbW"] = np.sqrt( selev.top_cand.xW**2 + selev.top_cand.xbW**2 )

            #
            # after minimizing, the ttbar distribution is centered around ~(0.5, 0.25) with surfaces of constant density approximiately constant radii
            #
            selev["top_cand", "rWbW"] = np.sqrt( (selev.top_cand.xbW - 0.25) ** 2 + (selev.top_cand.xW - 0.5) ** 2 )

            selev["xbW_reco"] = selev.top_cand.xbW
            selev["xW_reco"] = selev.top_cand.xW

            if "xbW" in selev.fields:  #### AGE: this should be temporary
                selev["delta_xbW"] = selev.xbW - selev.xbW_reco
                selev["delta_xW"] = selev.xW - selev.xW_reco


        if self.apply_FvT:
            quadJet["FvT_q_score"] = np.concatenate( ( np.reshape(np.array(selev.FvT.q_1234), (-1, 1)),
                                                       np.reshape(np.array(selev.FvT.q_1324), (-1, 1)),
                                                       np.reshape(np.array(selev.FvT.q_1423), (-1, 1)), ),
                                                     axis=1, )

        if self.run_SvB:

            if (self.classifier_SvB is not None) | (self.classifier_SvB_MA is not None):

                if "xbW_reco" not in selev.fields:
                    selev["xbW_reco"] = selev["xbW"]
                    selev["xW_reco"]  = selev["xW"]

                self.compute_SvB(selev)

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
            selev = selev[~(selev["quadJet_selected"].SR & selev.fourTag)]

        #
        # CutFlow
        #
        logging.debug(f"final weight {weights.weight()[:10]}")
        selev["weight"] = weights.weight()[selections.all(*allcuts)]
        if not shift_name:
            self._cutFlow.fill("passPreSel", selev, allTag=True)
            self._cutFlow.fill("passFourTag", selev )
            self._cutFlow.fill("pass0OthJets",selev )
            self._cutFlow.fill("passPreSel_woTrig", selev, allTag=True,
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)] ))
            self._cutFlow.fill("passDiJetMass", selev[selev.passDiJetMass])
            self._cutFlow.fill("passDiJetMass_woTrig", selev[selev.passDiJetMass],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)][selev.passDiJetMass] ))
            if self.run_SvB:
                selev['passSR'] = selev.passDiJetMass & selev["quadJet_selected"].SR
                self._cutFlow.fill( "SR", selev[selev.passSR] )
                self._cutFlow.fill( "SR_woTrig", selev[selev.passSR],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)][selev.passSR] ))
                selev['passSB'] = selev.passDiJetMass & selev["quadJet_selected"].SB
                self._cutFlow.fill( "SB", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)] )
                self._cutFlow.fill( "SB_woTrig", selev[(selev.passDiJetMass & selev["quadJet_selected"].SB)],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)][selev.passSB] ))
                self._cutFlow.fill("passSvB", selev[selev.passSvB])
                self._cutFlow.fill("passSvB_woTrig", selev[selev.passSvB],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)][selev.passSvB] ))
                self._cutFlow.fill("failSvB", selev[selev.failSvB])
                self._cutFlow.fill("failSvB_woTrig", selev[selev.failSvB],
                               wOverride=np.sum(weights.partial_weight(exclude=['CMS_bbbb_resolved_ggf_triggerEffSF'])[selections.all(*allcuts)][selev.failSvB] ))

            self._cutFlow.addOutput(processOutput, event.metadata["dataset"])

        #
        # Hists
        #

        if not self.run_systematics:

            fill = Fill(process=processName, year=year, weight="weight")

            hist = Collection( process=[processName],
                               year=[year],
                               tag=[3, 4, 0],  # 3 / 4/ Other
                               region=[2, 1, 0],  # SR / SB / Other
                               **dict((s, ...) for s in self.histCuts)
                               )

            #
            # To Add
            #

            #    m4j_vs_leadSt_dR = dir.make<TH2F>("m4j_vs_leadSt_dR", (name+"/m4j_vs_leadSt_dR; m_{4j} [GeV]; S_{T} leading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);
            #    m4j_vs_sublSt_dR = dir.make<TH2F>("m4j_vs_sublSt_dR", (name+"/m4j_vs_sublSt_dR; m_{4j} [GeV]; S_{T} subleading boson candidate #DeltaR(j,j); Entries").c_str(), 40,100,1100, 25,0,5);

            fill += hist.add( "nPVs", (101, -0.5, 100.5, ("PV.npvs", "Number of Primary Vertices")) )
            fill += hist.add( "nPVsGood", (101, -0.5, 100.5, ("PV.npvsGood", "Number of Good Primary Vertices")), )

            #
            #  Make classifier hists
            #
            if self.apply_FvT:
                FvT_skip = []
                if isMixedData or isDataForMixed or isTTForMixed:
                    FvT_skip = ["pt", "pm3", "pm4"]

            if "xbW" in selev.fields:  ### AGE: this should be temporary
                fill += hist.add("xW", (100, 0, 12, ("xW", "xW")))
                #fill += hist.add("delta_xW", (100, -5, 5, ("delta_xW", "delta xW")))
                #fill += hist.add("delta_xW_l", (100, -15, 15, ("delta_xW", "delta xW")))

            #
            # Separate reweighting for the different mixed samples
            #
            if isDataForMixed:
                for _FvT_name in event.metadata["FvT_names"]:
                    fill += SvBHists( (f"SvB_{_FvT_name}",    "SvB Classifier"),    "SvB",    weight=f"weight_{_FvT_name}" )
                    fill += SvBHists( (f"SvB_MA_{_FvT_name}", "SvB MA Classifier"), "SvB_MA", weight=f"weight_{_FvT_name}" )

            #
            # Jets
            #
            fill += Jet.plot(("selJets", "Selected Jets"),        "selJet",           skip=["deepjet_c"])
            fill += Jet.plot(("canJets", "Higgs Candidate Jets"), "canJet",           skip=["deepjet_c"])
            fill += Jet.plot(("othJets", "Other Jets"),           "notCanJet_coffea", skip=["deepjet_c"])
            fill += Jet.plot(("tagJets", "Tag Jets"),             "tagJet",           skip=["deepjet_c"])

            #
            #  Make quad jet hists
            #
            fill += LorentzVector.plot_pair( ("v4j", R"$HH_{4b}$"), "v4j", skip=["n", "dr", "dphi", "st"], bins={"mass": (120, 0, 1200)}, )
            fill += QuadJetHists( ("quadJet_selected", "Selected Quad Jet"), "quadJet_selected" )
            fill += QuadJetHists( ("quadJet_min_dr", "Min dR Quad Jet"), "quadJet_min_dr" )

            #
            #  Make classifier hists
            #
            if self.apply_FvT:
                FvT_skip = []
                if isMixedData or isDataForMixed or isTTForMixed:
                    FvT_skip = ["pt", "pm3", "pm4"]

                fill += FvTHists(("FvT", "FvT Classifier"), "FvT", skip=FvT_skip)

                fill += hist.add("quadJet_selected_FvT_score", (100, 0, 1, ("quadJet_selected.FvT_q_score", "Selected Quad Jet Diboson FvT q score") ) )
                fill += hist.add("quadJet_min_FvT_score",      (100, 0, 1, ("quadJet_min_dr.FvT_q_score",   "Min dR Quad Jet Diboson FvT q score"  ) ) )

                if self.JCM:
                    fill += hist.add("FvT_noFvT", (100, 0, 5, ("FvT.FvT", "FvT reweight")), weight="weight_noFvT")

            skip_all_but_n = ["deepjet_b", "energy", "eta", "id_jet", "id_pileup", "mass", "phi", "pt", "pz", "deepjet_c", ]

            fill += Jet.plot( ("selJets_noJCM", "Selected Jets"),        "selJet",       weight="weight_noJCM_noFvT", skip=skip_all_but_n, )
            fill += Jet.plot( ("tagJets_noJCM", "Tag Jets"),             "tagJet",       weight="weight_noJCM_noFvT", skip=skip_all_but_n, )
            fill += Jet.plot( ("tagJets_loose_noJCM", "Loose Tag Jets"), "tagJet_loose", weight="weight_noJCM_noFvT", skip=skip_all_but_n, )

            for iJ in range(4):
                fill += Jet.plot( (f"canJet{iJ}", f"Higgs Candidate Jets {iJ}"), f"canJet{iJ}", skip=["n", "deepjet_c"], )

            #
            #  Leptons
            #
            skip_muons = ["charge"] + Muon.skip_detailed_plots
            if not isMC:
                skip_muons += ["genPartFlav"]
            fill += Muon.plot( ("selMuons", "Selected Muons"), "selMuon", skip=skip_muons )

            if not isMixedData:
                skip_elecs = ["charge"] + Elec.skip_detailed_plots
                if not isMC:
                    skip_elecs += ["genPartFlav"]
                fill += Elec.plot( ("selElecs", "Selected Elecs"), "selElec", skip=skip_elecs )

            #
            # Top Candidates
            #
            if self.top_reconstruction in ["slow","fast"]:
                fill += TopCandHists(("top_cand", "Top Candidate"), "top_cand")

            if self.run_SvB:

                fill += SvBHists(("SvB",    "SvB Classifier"),    "SvB")
                fill += SvBHists(("SvB_MA", "SvB MA Classifier"), "SvB_MA")
                fill += hist.add( "quadJet_selected_SvB_q_score", ( 100, 0, 1, ( "quadJet_selected.SvB_q_score",  "Selected Quad Jet Diboson SvB q score") ) )
                fill += hist.add( "quadJet_min_SvB_MA_q_score",   ( 100, 0, 1, ( "quadJet_min_dr.SvB_MA_q_score", "Min dR Quad Jet Diboson SvB MA q score") ) )
                if isDataForMixed:
                    for _FvT_name in event.metadata["FvT_names"]:
                        fill += SvBHists( (f"SvB_{_FvT_name}",    "SvB Classifier"),    "SvB",    weight=f"weight_{_FvT_name}", )
                        fill += SvBHists( (f"SvB_MA_{_FvT_name}", "SvB MA Classifier"), "SvB_MA", weight=f"weight_{_FvT_name}", )

            #
            # fill histograms
            #
            # fill.cache(selev)
            fill(selev, hist)

            garbage = gc.collect()
            # print('Garbage:',garbage)

            friends = {}
            if self.make_classifier_input is not None:
                for k in ["ZZSR", "ZHSR", "HHSR", "SR", "SB"]:
                    selev[k] = selev["quadJet_selected"][k]

                selev["nSelJets"] = ak.num(selev.selJet)

                if "xbW_reco" in selev.fields:  #### AGE: this should be temporary
                    selev["xbW"] = selev["xbW_reco"]
                    selev["xW"]  = selev["xW_reco"]

                ####
                from ..helpers.classifier.HCR import dump_input_friend, dump_JCM_weight

                friends["friends"] = dump_input_friend( selev, self.make_classifier_input, "HCR_input", *selections, weight="weight" if isMC else "weight_noJCM_noFvT", NotCanJet="notCanJet_coffea") | dump_JCM_weight( selev, self.make_classifier_input, "JCM_weight", *selections, )

            output = hist.output | processOutput | friends
        #
        # Run systematics
        #
        else:

            shift_name = "nominal" if not shift_name else shift_name
            hist_SvB = Collection( process=[processName],
                                   year=[year],
                                   variation=[shift_name],
                                   tag=[4],  # 3 / 4/ Other
                                   region=[2],  # SR / SB / Other
                                   **dict((s, ...) for s in self.histCuts),
                                   )

            fill_SvB = Fill( process=processName, year=year, variation=shift_name, weight="weight" )
            fill_SvB += SvBHists(("SvB",    "SvB Classifier"),    "SvB",    skip=["ps", "ptt"])
            fill_SvB += SvBHists(("SvB_MA", "SvB MA Classifier"), "SvB_MA", skip=["ps", "ptt"])

            fill_SvB(selev, hist_SvB)

            if "nominal" in shift_name:
                logging.info(f"Weight variations {weights.variations}")

                dict_hist_SvB = {}
                for ivar in list(weights.variations):

                    dict_hist_SvB[ivar] = Collection( process=[processName],
                                                      year=[year],
                                                      variation=[ivar],
                                                      tag=[4],  # 3 / 4/ Other
                                                      region=[2],  # SR / SB / Other
                                                      **dict((s, ...) for s in self.histCuts) )


                    selev[f"weight_{ivar}"] = weights.weight(modifier=ivar)[ selections.all(*allcuts) ]
                    fill_SvB_ivar = Fill( process=processName, year=year, variation=ivar, weight=f"weight_{ivar}", )

                    logging.debug(f"{ivar} {selev['weight']}")

                    fill_SvB_ivar += SvBHists( ("SvB",    "SvB Classifier"),    "SvB",    skip=["ps", "ptt"] )
                    fill_SvB_ivar += SvBHists( ("SvB_MA", "SvB MA Classifier"), "SvB_MA", skip=["ps", "ptt"] )

                    fill_SvB_ivar(selev, dict_hist_SvB[ivar])

                    for ih in hist_SvB.output["hists"].keys():
                        hist_SvB.output["hists"][ih] = ( hist_SvB.output["hists"][ih] + dict_hist_SvB[ivar].output["hists"][ih] )

            output = hist_SvB.output | processOutput

        return output

        #
        # Done
        #
        elapsed = time.time() - tstart
        logging.debug(f"{chunk}{nEvent/elapsed:,.0f} events/s")

    def compute_SvB(self, event):
        import torch
        import torch.nn.functional as F

        n = len(event)

        j = torch.zeros(n, 4, 4)
        j[:, 0, :] = torch.tensor(event.canJet.pt)
        j[:, 1, :] = torch.tensor(event.canJet.eta)
        j[:, 2, :] = torch.tensor(event.canJet.phi)
        j[:, 3, :] = torch.tensor(event.canJet.mass)

        o = torch.zeros(n, 5, 8)
        o[:, 0, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.pt,       target=8, clip=True) ), -1, ) )
        o[:, 1, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.eta,      target=8, clip=True) ), -1, ) )
        o[:, 2, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.phi,      target=8, clip=True) ), -1, ) )
        o[:, 3, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.mass,     target=8, clip=True) ), -1, ) )
        o[:, 4, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.isSelJet, target=8, clip=True) ), -1, ) )

        a = torch.zeros(n, 4)
        a[:, 0] = float(event.metadata["year"][3])
        a[:, 1] = torch.tensor(event.nJet_selected)
        a[:, 2] = torch.tensor(event.xW_reco)
        a[:, 3] = torch.tensor(event.xbW_reco)

        e = torch.tensor(event.event) % 3

        for classifier in ["SvB", "SvB_MA"]:

            if classifier == "SvB":
                c_logits, q_logits = self.classifier_SvB(j, o, a, e)

            if classifier == "SvB_MA":
                c_logits, q_logits = self.classifier_SvB_MA(j, o, a, e)

            c_score, q_score = ( F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy(), )

            # classes = [mj,tt,zz,zh,hh]
            SvB = ak.zip( { "pmj": c_score[:, 0],
                            "ptt": c_score[:, 1],
                            "pzz": c_score[:, 2],
                            "pzh": c_score[:, 3],
                            "phh": c_score[:, 4],
                            "q_1234": q_score[:, 0],
                            "q_1324": q_score[:, 1],
                            "q_1423": q_score[:, 2],
                            }
                          )

            largest_name = np.array(["None", "ZZ", "ZH", "HH"])
            SvB["ps"] = SvB.pzz + SvB.pzh + SvB.phh
            SvB["passMinPs"] = (SvB.pzz > 0.01) | (SvB.pzh > 0.01) | (SvB.phh > 0.01)
            SvB["zz"] = (SvB.pzz > SvB.pzh) & (SvB.pzz > SvB.phh)
            SvB["zh"] = (SvB.pzh > SvB.pzz) & (SvB.pzh > SvB.phh)
            SvB["hh"] = (SvB.phh > SvB.pzz) & (SvB.phh > SvB.pzh)
            SvB["largest"] = largest_name[ SvB.passMinPs * ( 1 * SvB.zz + 2* SvB.zh + 3*SvB.hh ) ]

            this_ps_zz = np.full(len(event), -1, dtype=float)
            this_ps_zz[ SvB.zz ] = SvB.ps[ SvB.zz ]
            this_ps_zz[ SvB.passMinPs == False ] = -2
            SvB['ps_zz'] = this_ps_zz

            this_ps_zh = np.full(len(event), -1, dtype=float)
            this_ps_zh[ SvB.zh ] = SvB.ps[ SvB.zh ]
            this_ps_zh[ SvB.passMinPs == False ] = -2
            SvB['ps_zh'] = this_ps_zh

            this_ps_hh = np.full(len(event), -1, dtype=float)
            this_ps_hh[ SvB.hh ] = SvB.ps[ SvB.hh ]
            this_ps_hh[ SvB.passMinPs == False ] = -2
            SvB['ps_hh'] = this_ps_hh

            if classifier in event.fields:
                error = ~np.isclose(event[classifier].ps, SvB.ps, atol=1e-5, rtol=1e-3)
                if np.any(error):
                    delta = np.abs(event[classifier].ps - SvB.ps)
                    worst = np.max(delta) == delta
                    worst_event = event[worst][0]

                    logging.warning( f"WARNING: Calculated {classifier} does not agree within tolerance for some events ({np.sum(error)}/{len(error)}) {delta[worst]}" )

                    logging.warning("----------")

                    for field in event[classifier].fields:
                        logging.warning(f"{field} {worst_event[classifier][field]}")

                    logging.warning("----------")

                    for field in SvB.fields:
                        logging.warning(f"{field} {SvB[worst][field]}")

            # del event[classifier]
            event[classifier] = SvB

    def postprocess(self, accumulator):
        return accumulator
