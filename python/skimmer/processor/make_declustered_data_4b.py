import yaml
from skimmer.processor.picoaod import PicoAOD, fetch_metadata, resize
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b
from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from analysis.helpers.common import init_jet_factory
from copy import copy
import logging
import awkward as ak

class DeClusterer(PicoAOD):
    def __init__(self, loosePtForSkim=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loosePtForSkim = loosePtForSkim
        self.corrections_metadata = yaml.safe_load(open('analysis/metadata/corrections.yml', 'r'))
        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult_lowpt_forskim",
            "passJetMult",
            "passPreSel_lowpt_forskim",
            "passPreSel",
        ]

    def select(self, event):

        isMC    = True if event.run[0] == 1 else False
        year    = event.metadata['year']
        dataset = event.metadata['dataset']

        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )

        juncWS = [ self.corrections_metadata[year]["JERC"][0].replace("STEP", istep)
                   for istep in ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"] ] #+ self.corrections_metadata[year]["JERC"][2:]

        #old_jets = copy(event.Jet)
        jets = init_jet_factory(juncWS, event, isMC)
        event["Jet"] = jets

        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year], loosePtForSkim=self.loosePtForSkim  )

        weights = Weights(len(event), storeIndividual=True)

        #
        # general event weights
        #
        if isMC:
            weights.add( "genweight_", event.genWeight )
            weights.add( "CMS_bbbb_resolved_ggf_triggerEffSF", event.trigWeight.Data, event.trigWeight.MC, ak.where(event.passHLT, 1.0, 0.0), )
            list_weight_names.append('CMS_bbbb_resolved_ggf_triggerEffSF')
            logging.debug( f"trigWeight {weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[:10]}\n" )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( np.full(len(event), True) if isMC else event.passHLT ) )
        if self.loosePtForSkim:
            selections.add( 'passJetMult_lowpt_forskim', event.passJetMult_lowpt_forskim )
        selections.add( 'passJetMult',   event.passJetMult )
        if self.loosePtForSkim:
            selections.add( "passPreSel_lowpt_forskim",  event.passPreSel_lowpt_forskim)
        selections.add( 'passJetMult', event.passJetMult )
        selections.add( "passPreSel",  event.passPreSel)
        selections.add( "passFourTag", event.fourTag)


        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        if self.loosePtForSkim:
            all_cuts = ["passNoiseFilter", "passHLT", "passJetMult_lowpt_forskim", "passJetMult", "passPreSel_lowpt_forskim", "passPreSel"]
        else:
            all_cuts = ["passNoiseFilter", "passHLT", "passJetMult", "passPreSel", "passFourTag"]

        for cut in all_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        if self.loosePtForSkim:
            selection = event.lumimask & event.passNoiseFilter & event.passJetMult_lowpt_forskim & event.passPreSel_lowpt_forskim
        else:
            selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.passPreSel & event.fourTag
        if not isMC: selection = selection & event.passHLT

        selev = event[selections.all(*cumulative_cuts)]
        #selev = event[selection]

        n_jet = ak.num(selev.Jet)
        total_jet = int(ak.sum(n_jet))


        branches = ak.Array(
            {
                # replace branch by using the same name as nanoAOD
                "Jet_pt": selev.event % 53 + ak.local_index(selev.Jet.pt),
                # replace another branch
                "Jet_phi": ak.unflatten(np.zeros(total_jet), n_jet),
                # create new branch
                "Jet_isGood": ak.unflatten(
                    np.repeat(selev.event % 2 == 0, n_jet), n_jet
                ),
                # create new regular branch
                "isBad": selev.event % 2 == 1,
            }
        )

        result = {"total_jet": total_jet}

        return (selection,
                branches,
                result,
                )
