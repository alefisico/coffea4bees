import yaml
from skimmer.processor.picoaod import PicoAOD, fetch_metadata, resize
from analysis.helpers.selection_basic_4b import apply_event_selection_4b, apply_object_selection_4b
from coffea.nanoevents import NanoEventsFactory

from jet_clustering.clustering   import cluster_bs
from jet_clustering.declustering import make_synthetic_event, clean_ISR
from analysis.helpers.SvB_helpers import setSvBVars, subtract_ttbar_with_SvB
from analysis.helpers.FriendTreeSchema import FriendTreeSchema
from base_class.math.random import Squares
from analysis.helpers.event_weights import add_weights

from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from analysis.helpers.common import init_jet_factory, apply_btag_sf
from copy import copy
import logging
import awkward as ak
import uproot

class DeClusterer(PicoAOD):
    def __init__(self, clustering_pdfs_file = "None", subtract_ttbar_with_weights = False, declustering_rand_seed=5, *args, **kwargs):
        kwargs["pico_base_name"] = f'picoAOD_seed{declustering_rand_seed}'
        super().__init__(*args, **kwargs)

        logging.info(f"\nRunning Declusterer with these parameters: clustering_pdfs_file = {clustering_pdfs_file}, subtract_ttbar_with_weights = {subtract_ttbar_with_weights}, declustering_rand_seed = {declustering_rand_seed}, args = {args}, kwargs = {kwargs}")

        if not clustering_pdfs_file == "None":
            self.clustering_pdfs = yaml.safe_load(open(clustering_pdfs_file, "r"))
            logging.info(f"Loaded {len(self.clustering_pdfs.keys())} PDFs from {clustering_pdfs_file}")
        else:
            self.clustering_pdfs = None

        self.subtract_ttbar_with_weights = subtract_ttbar_with_weights
        self.declustering_rand_seed = declustering_rand_seed
        self.corrections_metadata = yaml.safe_load(open('analysis/metadata/corrections.yml', 'r'))
        self.cutFlowCuts = [
            "all",
            "passHLT",
            "passNoiseFilter",
            "passJetMult",
            "passFourTag",
            "pass_ttbar_filter",
        ]

        self.skip_collections = kwargs["skip_collections"]
        self.skip_branches    = kwargs["skip_branches"]


    def select(self, event):

        isMC    = True if event.run[0] == 1 else False
        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        fname   = event.metadata['filename']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        nEvent = len(event)
        year_label = self.corrections_metadata[year]['year_label']

        path = fname.replace(fname.split("/")[-1], "")

        if self.subtract_ttbar_with_weights:

            SvB_MA_file = f'{fname.replace("picoAOD", "SvB_MA_ULHH")}'
            event["SvB_MA"] = ( NanoEventsFactory.from_root( SvB_MA_file,
                                                             entry_start=estart, entry_stop=estop, schemaclass=FriendTreeSchema ).events().SvB_MA )

            if not ak.all(event.SvB_MA.event == event.event):
                raise ValueError("ERROR: SvB_MA events do not match events ttree")

            # defining SvB_MA
            setSvBVars("SvB_MA", event)


        event = apply_event_selection_4b( event, isMC, self.corrections_metadata[year] )

        ## adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, isMC, dataset, year_label,
                                                  estart, estop,
                                                  self.corrections_metadata[year],
                                                  apply_trigWeight = True,
                                                  isTTForMixed=False,
                                                 )

        event = apply_object_selection_4b( event, self.corrections_metadata[year]  )


        #weights = Weights(len(event), storeIndividual=True)

        #
        # Get the trigger weights
        #
        if isMC:
            if "GluGlu" in dataset:
                ### this is temporary until trigWeight is computed in new code
                trigWeight_file = uproot.open(f'{event.metadata["filename"].replace("picoAOD", "trigWeights")}')['Events']
                trigWeight = trigWeight_file.arrays(['event', 'trigWeight_Data', 'trigWeight_MC'], entry_start=estart,entry_stop=estop)
                if not ak.all(trigWeight.event == event.event):
                    raise ValueError('trigWeight events do not match events ttree')

                event["trigWeight_Data"] = trigWeight["trigWeight_Data"]
                event["trigWeight_MC"]   = trigWeight["trigWeight_MC"]


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

        #
        # Add Btag SF
        #
        if isMC:
            weights.add( "CMS_btag",
                         apply_btag_sf( event.selJet, correction_file=self.corrections_metadata[year]["btagSF"], btag_uncertainties=None, )["btagSF_central"], )
            list_weight_names.append(f"CMS_btag")

            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )
            event["weight"] = weights.weight()



        selection = event.lumimask & event.passNoiseFilter & event.passJetMult & event.fourTag
        if not isMC: selection = selection & event.passHLT

        selev = event[selections.all(*cumulative_cuts)]

        #
        #  TTbar subtractions using weights
        #
        if self.subtract_ttbar_with_weights:

            pass_ttbar_filter_selev = subtract_ttbar_with_SvB(selev, dataset, year)

            pass_ttbar_filter = np.full( len(event), True)
            pass_ttbar_filter[ selections.all(*cumulative_cuts) ] = pass_ttbar_filter_selev
            selections.add( 'pass_ttbar_filter', pass_ttbar_filter )
            cumulative_cuts.append("pass_ttbar_filter")
            self._cutFlow.fill( "pass_ttbar_filter", event[selections.all(*cumulative_cuts)], allTag=True )

            selection = selection & pass_ttbar_filter
            selev = selev[pass_ttbar_filter_selev]

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
        declustered_jets = make_synthetic_event(clustered_jets, self.clustering_pdfs, declustering_rand_seed=self.declustering_rand_seed)

        declustered_jets = declustered_jets[ak.argsort(declustered_jets.pt, axis=1, ascending=False)]

        n_jet = ak.num(declustered_jets)
        total_jet = int(ak.sum(n_jet))


        out_branches = {
                # Update jets with new kinematics
                "Jet_pt":              declustered_jets.pt, #ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_eta":             declustered_jets.eta,
                "Jet_phi":             declustered_jets.phi,
                "Jet_mass":            declustered_jets.mass,
                "Jet_btagDeepFlavB":   declustered_jets.btagDeepFlavB,
                "Jet_jet_flavor_bit":  declustered_jets.jet_flavor_bit,
                "Jet_jetId":           ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_puId":            ak.unflatten(np.full(total_jet, 7), n_jet),
                "Jet_bRegCorr":        ak.unflatten(np.full(total_jet, 1), n_jet),
                # create new regular branch
                "nClusteredJets":      selev.nClusteredJets,
            }

        if isMC:
            out_branches["trigWeight_Data"] = selev.trigWeight_Data
            out_branches["trigWeight_MC"]   = selev.trigWeight_MC
            out_branches["CMSbtag"]        = weights.partial_weight(include=["CMS_btag"])[selections.all(*cumulative_cuts)]

        #
        #  Need to skip all the other jet branches to make sure they have the same number of jets
        #
        for f in event.Jet.fields:
            bname = f"Jet_{f}"
            if bname not in out_branches:
                self.skip_branches.append(bname)

        self.update_branch_filter(self.skip_collections, self.skip_branches)
        branches = ak.Array(out_branches)

        result = {"total_jet": total_jet}

        return (selection,
                branches,
                result,
                )
