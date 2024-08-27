import yaml
from skimmer.processor.picoaod import PicoAOD, fetch_metadata, resize
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b

from jet_clustering.clustering   import cluster_bs
from jet_clustering.declustering import make_synthetic_event, clean_ISR

from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from analysis.helpers.common import init_jet_factory
from copy import copy
import logging
import awkward as ak

class DeClusterer(PicoAOD):
    def __init__(self, clustering_pdfs_file = "None", *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not clustering_pdfs_file == "None":
            self.clustering_pdfs = yaml.safe_load(open(clustering_pdfs_file, "r"))
            logging.info(f"Loaded {len(self.clustering_pdfs.keys())} PDFs from {clustering_pdfs_file}")
        else:
            self.clustering_pdfs = None

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
        nEvent = len(event)

        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )

        juncWS = [ self.corrections_metadata[year]["JERC"][0].replace("STEP", istep)
                   for istep in ["L1FastJet", "L2Relative", "L2L3Residual", "L3Absolute"] ] #+ self.corrections_metadata[year]["JERC"][2:]

        #old_jets = copy(event.Jet)
        jets = init_jet_factory(juncWS, event, isMC)
        event["Jet"] = jets

        event = apply_object_selection_4b( event, year, isMC, dataset, self.corrections_metadata[year]  )

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
        selections.add( 'passJetMult',   event.passJetMult )
        selections.add( "passFourTag", event.fourTag)

        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        all_cuts = ["passNoiseFilter", "passHLT", "passJetMult","passFourTag"]

        for cut in all_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )

        selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.fourTag
        if not isMC: selection = selection & event.passHLT

        selev = event[selections.all(*cumulative_cuts)]

        # ## TTbar subtractions
        # if self.subtract_ttbar_with_weights:
        #     ttbar_rand = np.random.uniform(low=0, high=1.0, size=len(selev))
        #     pass_ttbar_filter = np.full( len(event), True)
        #     pass_ttbar_filter[ selections.all(*allcuts) ] = (ttbar_rand > selev.original_SvB_MA.tt_vs_mj)
        #     selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
        #     allcuts.append("pass_ttbar_filter")
        #     selev = selev[(ttbar_rand > selev.original_SvB_MA.tt_vs_mj)]

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
        #canJet["puId"] = selev.Jet.puId[canJet_idx]
        #canJet["jetId"] = selev.Jet.puId[canJet_idx]
        if isMC:
            canJet["hadronFlavour"] = selev.Jet.hadronFlavour[canJet_idx]

        canJet["calibration"] = selev.Jet.calibration[canJet_idx]

        #
        # pt sort canJets
        #
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]

        notCanJet = selev.Jet[notCanJet_idx]
        notCanJet = notCanJet[notCanJet.selected_loose]
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        #
        # Do the Clustering
        #
        canJet["jet_flavor"] = "b"
        notCanJet["jet_flavor"] = "j"

        jets_for_clustering = ak.concatenate([canJet, notCanJet], axis=1)
        jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]

        clustered_jets, _clustered_splittings = cluster_bs(jets_for_clustering, debug=False)
        clustered_jets = clean_ISR(clustered_jets, _clustered_splittings)

        mask_unclustered_jet = (clustered_jets.jet_flavor == "b") | (clustered_jets.jet_flavor == "j")
        selev["nClusteredJets"] = ak.num(clustered_jets[~mask_unclustered_jet])

        #
        # Declustering
        #
        declustered_jets = make_synthetic_event(clustered_jets, self.clustering_pdfs)


        canJet = declustered_jets[declustered_jets.jet_flavor == "b"]
        btag_rand = np.random.uniform(low=0.6, high=1.0, size=len(ak.flatten(canJet,axis=1)))
        canJet["btagDeepFlavB"] = ak.unflatten(btag_rand, ak.num(canJet))
        # Set these pt > 40 and |eta| < 2.4
        canJet = canJet[ak.argsort(canJet.pt, axis=1, ascending=False)]

        notCanJet = declustered_jets[declustered_jets.jet_flavor == "j"]
        btag_rand = np.random.uniform(low=0.0, high=0.6, size=len(ak.flatten(notCanJet,axis=1)))
        notCanJet["btagDeepFlavB"] = ak.unflatten(btag_rand, ak.num(notCanJet))
        notCanJet = notCanJet[ak.argsort(notCanJet.pt, axis=1, ascending=False)]

        new_jets = ak.concatenate([canJet, notCanJet], axis=1)
        new_jets = new_jets[ak.argsort(new_jets.pt, axis=1, ascending=False)]

        # n_jet_old_all = ak.num(selev.Jet)
        # total_jet_old_all = int(ak.sum(n_jet_old_all))
        # selev.Jet = selev.Jet[selev.Jet.selected_loose]
        #
        # n_jet_old = ak.num(selev.Jet)
        # total_jet_old = int(ak.sum(n_jet_old))

        n_jet = ak.num(new_jets)
        total_jet = int(ak.sum(n_jet))

        # delta_njet = n_jet - n_jet_old
        # delta_njet_total = total_jet - total_jet_old
        # print(f"delta_njet_total is {delta_njet_total} {total_jet} {total_jet_old} {total_jet_old_all} \n")
        # print(f"delta_njet  {ak.any(delta_njet)} {delta_njet[~(delta_njet ==0)]}   \n")

        #'isGood', 'btagDeepB', 'cleanmask', 'jetId', 'area', 'chEmEF', 'eta', 'pt', 'bRegCorr', 'rawFactor', 'btagDeepFlavB', 'puId', 'phi', 'neEmEF', 'btagCSVV2', 'mass'
        branches = ak.Array(
            {
                # Update jets with new kinematics
                "Jet_pt":              new_jets.pt, #ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_eta":             new_jets.eta,
                "Jet_phi":             new_jets.phi,
                "Jet_mass":            new_jets.mass,
                "Jet_btagDeepFlavB":   new_jets.btagDeepFlavB,
                # These throw error
                #"Jet_jet_flavor":      ak.unflatten(np.full(total_jet, "b".encode("utf-8")), n_jet), # jets.jet_flavor ,
                #"Jet_jet_flavor":      ak.unflatten(np.full(total_jet, "b" n_jet), # jets.jet_flavor ,
                # Set to dummy values for synthetic jets
                "Jet_jetId":     ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_puId":      ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_bRegCorr":  ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_isGood":    ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_btagDeepB": ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_cleanmask": ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_area":      ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_chEmEF":    ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_rawFactor": ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_neEmEF":    ak.unflatten(np.full(total_jet, 1), n_jet),
                "Jet_btagCSVV2": ak.unflatten(np.full(total_jet, 1), n_jet),

                # create new regular branch
                "nClusteredJets": selev["nClusteredJets"],
            }
        )

        result = {"total_jet": total_jet}

        return (selection,
                branches,
                result,
                )
