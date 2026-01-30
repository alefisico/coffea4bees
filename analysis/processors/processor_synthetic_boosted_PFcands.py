import time
import awkward as ak
import numpy as np
import warnings
import yaml
import logging
from collections import OrderedDict

from coffea import processor
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents import NanoAODSchema
import hist
import vector
from coffea.nanoevents.methods import vector as coffea_vector

from src.hist_tools import Template
from src.hist_tools import Collection, Fill
from src.hist_tools.object import LorentzVector, Jet, PFCand
from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from src.physics.event_selection import apply_event_selection

from coffea4bees.jet_clustering.clustering import comb_jet_flavor
from coffea4bees.jet_clustering.declustering import make_synthetic_event
from coffea4bees.jet_clustering.clustering_hist_templates import (
    ClusterHistsBoosted,
    ClusterHistsDetailedBoosted,
)
from coffea4bees.jet_clustering.declustering import (
    compute_decluster_variables,
    get_splitting_name,
    get_list_of_combined_jet_types,
    get_list_of_all_sub_splittings
)

# Setup
NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")


class analysis(processor.ProcessorABC):
    def __init__(
        self,
        *,
        corrections_metadata: dict = None,
        clustering_pdfs_file: str = "coffea4bees/jet_clustering/jet-splitting-PDFs-boosted-00-03-00/clustering_pdfs_vs_pT_XXX.yml",
        declustering_rand_seed: int = 5,
        declustering_config: str = "coffea4bees/skimmer/metadata/declustering_boosted.yml",
        **kwargs,
    ):
        logging.debug("\nInitialize Analysis Processor")
        self.corrections_metadata = corrections_metadata

        # declustering inputs (same meaning as skimmer)
        self.clustering_pdfs_file = clustering_pdfs_file
        self.declustering_rand_seed = declustering_rand_seed
        self.declustering_config = declustering_config

    def process(self, event):
        tstart = time.time()
        fname = event.metadata["filename"]
        year = event.metadata["year"]
        dataset = event.metadata["dataset"]
        estart = event.metadata["entrystart"]
        estop = event.metadata["entrystop"]
        processName = event.metadata["processName"]
        chunk = f"{dataset}::{estart:6d}:{estop:6d} >>> "
        isMC = True if event.run[0] == 1 else False
        nEvent = len(event)

        logging.info(fname)
        logging.info(f"Process {nEvent} Events")

        clustering_pdfs_file = self.clustering_pdfs_file.replace("XXX", year)

        if clustering_pdfs_file not in [None, "None", ""]:
            with open(clustering_pdfs_file, "r") as f:
                clustering_pdfs = yaml.safe_load(f)
            logging.info(
                f"Loaded {len(clustering_pdfs.keys())} PDFs from {clustering_pdfs_file}"
            )
        else:
            clustering_pdfs = None

        # -------- Event-level selections --------
        event = apply_event_selection(
            event,
            self.corrections_metadata[year],
            cut_on_lumimask=True,
        )

        # Base fatjet object
        selFatJet = event.FatJet
        selFatJet = selFatJet[selFatJet.subJetIdx1 >= 0]
        selFatJet = selFatJet[selFatJet.subJetIdx2 >= 0]
        selFatJet = selFatJet[selFatJet.pt > 300]
        selFatJet = selFatJet[selFatJet.subJetIdx1 < 4]
        selFatJet = selFatJet[selFatJet.subJetIdx2 < 4]
        selFatJet = selFatJet[
            (selFatJet.subjets[:, :, 0] + selFatJet.subjets[:, :, 1]).pt > 300
        ]
        selFatJet = selFatJet[
            (selFatJet.subjets[:, :, 0] + selFatJet.subjets[:, :, 1]).mass > 50
        ]

        event["selFatJet"] = selFatJet
        event["passNFatJets"] = ak.num(event.selFatJet) == 2

        selections = PackedSelection()
        selections.add("lumimask", event.lumimask)
        selections.add("passNoiseFilter", event.passNoiseFilter)
        selections.add("passHLT", np.full(len(event), True) if isMC else event.passHLT)
        selections.add("passNFatJets", event.passNFatJets)

        event["weight"] = 1.0

        # -------- Cutflow --------
        sel_dict = OrderedDict(
            {
                "all": selections.require(lumimask=True),
                "passNoiseFilter": selections.require(
                    lumimask=True, passNoiseFilter=True
                ),
                "passHLT": selections.require(
                    lumimask=True, passNoiseFilter=True, passHLT=True
                ),
                "passNFatJets": selections.require(
                    lumimask=True, passNoiseFilter=True, passHLT=True, passNFatJets=True
                ),
            }
        )
        self.cutFlow = cutflow_4b()
        for cut, sel in sel_dict.items():
            self.cutFlow.fill(cut, event[sel], allTag=True)

        list_of_cuts = ["lumimask", "passNoiseFilter", "passHLT", "passNFatJets"]
        selev = event[selections.all(*list_of_cuts)]

        # -------- Sync fields for compatibility --------
        indices_str = []
        for arr in selev.selFatJet.pt:
            indices_str.append([f"({i},{i})" for i in range(len(arr))])
        selev["selFatJet", "btag_string"] = indices_str
        # CMS NanoAOD field present in our data:
        selev["selFatJet", "btagScore"] = selev.selFatJet.particleNet_XbbVsQCD

        # -------- Build fatjet container (i0/i1 sorted by pt) --------
        sorted_sub_jets = selev.selFatJet.subjets
        sorted_sub_jets = sorted_sub_jets[
            ak.argsort(sorted_sub_jets.pt, axis=2, ascending=False)
        ]

        if "tau1" not in sorted_sub_jets.fields:
            # Synthetic sometimes lacks tau{1,2,3}; mirror fields for plotting API
            sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau1")
            sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau2")
            sorted_sub_jets = ak.with_field(sorted_sub_jets, sorted_sub_jets.pt, "tau3")

        # PF ↔ FatJet mapping (kept simple/consistent with existing data)
        PFCandFatJet0_mask = (
            selev.FatJetPFCands.jetIdx == ak.local_index(selev.FatJet, axis=1)[:, 0]
        )
        PFCandFatJet1_mask = (
            selev.FatJetPFCands.jetIdx == ak.local_index(selev.FatJet, axis=1)[:, 1]
        )
        PFCandIndex_FatJet0 = selev.FatJetPFCands[PFCandFatJet0_mask].pFCandsIdx
        PFCandIndex_FatJet1 = selev.FatJetPFCands[PFCandFatJet1_mask].pFCandsIdx
        PFCands_perFatJet = ak.Array(
            [
                [a, b]
                for a, b in zip(
                    selev.PFCands[PFCandIndex_FatJet0],
                    selev.PFCands[PFCandIndex_FatJet1],
                )
            ]
        )
        # print(f" nPFCands for FatJet0 {ak.num(PFCandIndex_FatJet0)}\n")
        # print(f" nPFCands for FatJet1 {ak.num(PFCandIndex_FatJet1)}\n")
        #
        # print(f" PFCands for FatJet0 {selev.PFCands[PFCandIndex_FatJet0].pdgId.tolist()}\n")
        # print(f" PFCands for FatJet1 {selev.PFCands[PFCandIndex_FatJet1].pdgId.tolist()}\n")
        fatjets = ak.zip(
            {
                "p": sorted_sub_jets[:, :, 0] + sorted_sub_jets[:, :, 1],
                "i0": sorted_sub_jets[:, :, 0],
                "i1": sorted_sub_jets[:, :, 1],
                "PFCands": PFCands_perFatJet,
            },
            depth_limit=1,
        )

        # Dijet vars
        fatjets["p", "st"] = fatjets.i0.pt + fatjets.i1.pt
        fatjets["p", "dr"] = fatjets.i0.delta_r(fatjets.i1)
        fatjets["p", "dphi"] = fatjets.i0.delta_phi(fatjets.i1)

        # Groomed-shape helpers
        fatjets["i0", "tau21"] = fatjets.i0.tau2 / (fatjets.i0.tau1 + 1e-3)
        fatjets["i1", "tau21"] = fatjets.i1.tau2 / (fatjets.i1.tau1 + 1e-3)
        fatjets["i0", "tau32"] = fatjets.i0.tau3 / (fatjets.i0.tau2 + 1e-3)
        fatjets["i1", "tau32"] = fatjets.i1.tau3 / (fatjets.i1.tau2 + 1e-3)

        # Dummy subjet btag string
        fatjets["i0", "btag_string"] = "0.001"
        fatjets["i1", "btag_string"] = "0.001"

        # Subjet flavors and iA/iB ordering
        fatjets["i0", "jet_flavor"] = ak.where(fatjets.i0.tau21 > 0.5, "b", "bj")
        fatjets["i1", "jet_flavor"] = ak.where(fatjets.i1.tau21 > 0.5, "b", "bj")
        i1_more_complex = (fatjets.i1.jet_flavor == "bj") & (
            fatjets.i0.jet_flavor == "b"
        )
        fatjets["iA"] = ak.where(i1_more_complex, fatjets.i1, fatjets.i0)
        fatjets["iB"] = ak.where(i1_more_complex, fatjets.i0, fatjets.i1)

        # Combined fatjet flavor back to selFatJet
        C = [
            comb_jet_flavor(a, b)
            for a, b in zip(
                ak.flatten(fatjets.iA.jet_flavor), ak.flatten(fatjets.iB.jet_flavor)
            )
        ]
        fatjet_flavor_flat = np.array(C)
        selev["selFatJet", "jet_flavor"] = ak.unflatten(
            fatjet_flavor_flat, ak.num(selev.selFatJet)
        )

        # -------- Build combined splitting object (synthetic style) --------
        comb = ak.zip(
            {
                "pt": (fatjets.iA + fatjets.iB).pt,
                "eta": (fatjets.iA + fatjets.iB).eta,
                "phi": (fatjets.iA + fatjets.iB).phi,
                "mass": (fatjets.iA + fatjets.iB).mass,
                "jet_flavor": selev.selFatJet.jet_flavor,
                "btag_string": selev.selFatJet.btag_string,
                "part_A": ak.zip(
                    {
                        "pt": fatjets.iA.pt,
                        "eta": fatjets.iA.eta,
                        "phi": fatjets.iA.phi,
                        "mass": fatjets.iA.mass,
                        "jet_flavor": fatjets.iA.jet_flavor,
                        "btag_string": fatjets.iA.btag_string,
                        "tau21": fatjets.iA.tau21,
                        "tau32": fatjets.iA.tau32,
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.backends.awkward.behavior,
                ),
                "part_B": ak.zip(
                    {
                        "pt": fatjets.iB.pt,
                        "eta": fatjets.iB.eta,
                        "phi": fatjets.iB.phi,
                        "mass": fatjets.iB.mass,
                        "jet_flavor": fatjets.iB.jet_flavor,
                        "btag_string": fatjets.iB.btag_string,
                        "tau21": fatjets.iB.tau21,
                        "tau32": fatjets.iB.tau32,
                    },
                    with_name="PtEtaPhiMLorentzVector",
                    behavior=vector.backends.awkward.behavior,
                ),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.backends.awkward.behavior,
        )

        # Explicit masses for convenience
        comb = ak.with_field(comb, comb.part_A.mass, "mass_A")
        comb = ak.with_field(comb, comb.part_B.mass, "mass_B")
        comb = ak.with_field(comb, (comb.part_A + comb.part_B).mass, "mass_AB")

        # -------- Declustering --------
        compute_decluster_variables(comb)

        # -------- build clustered_jets  --------
        clustered_jets = ak.zip(
            {
                "pt":   ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).pt,   np.float64),
                "eta":  ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).eta,  np.float64),
                "phi":  ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).phi,  np.float64),
                "mass": ak.values_astype((selev.selFatJet.subjets[:, :, 0] + selev.selFatJet.subjets[:, :, 1]).mass, np.float64),
                "jet_flavor": selev.selFatJet.jet_flavor,
                "btag_string": selev.selFatJet.btag_string,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=coffea_vector.behavior,
        )

        # -------- Fake-on-real declustering (skimmer-style) --------
        declustered_jets = make_synthetic_event(
            clustered_jets,
            clustering_pdfs,
            declustering_rand_seed=self.declustering_rand_seed,
            b_pt_threshold=20,
            dr_threshold=0,
            chunk=chunk,
            debug=False,
            splitting_types_to_ignore=[("bj", "b")],
        )

        declustered_jets = declustered_jets[ak.argsort(declustered_jets.btagScore, axis=1, ascending=True)]

        # -------- keep only events where output matches (2 per fatjet) --------
        mask = ak.num(declustered_jets) == 2 * ak.num(selev.selFatJet)
        selev, fatjets, comb, declustered_jets = selev[mask], fatjets[mask], comb[mask], declustered_jets[mask]

        declustered_pairs = ak.unflatten(declustered_jets, 2, axis=1)

        i0_new, i1_new = declustered_pairs[:, :, 0], declustered_pairs[:, :, 1]

        fatjets = ak.with_field(fatjets, i0_new, "i0_new")
        fatjets = ak.with_field(fatjets, i1_new, "i1_new")
        fatjets = ak.with_field(fatjets, i0_new + i1_new, "p_new")

        # Splitting names
        split_name_flat = [get_splitting_name(str(i)) for i in ak.flatten(comb.jet_flavor)]
        split_name = ak.unflatten(split_name_flat, ak.num(comb))
        comb = ak.with_field(comb, split_name, "splitting_name")

        # Enumerate available splitting types
        cleaned_types = get_list_of_combined_jet_types(comb)
        all_splits = []
        for s in cleaned_types:
            all_splits += get_list_of_all_sub_splittings(s)
        cleaned_splitting_name = set(get_splitting_name(i) for i in all_splits)
        if "1b0j/0b1j" in cleaned_splitting_name:
            cleaned_splitting_name.remove("1b0j/0b1j")

        # Group by type
        for s_type in cleaned_splitting_name:
            selev[f"splitting_{s_type}"] = comb[comb.splitting_name == s_type]

        # By mass categories (use mass_AB we defined above)
        selev["splitting_1b0j/1b0j_lowMass"] = comb[comb.mass_AB < 75.0]
        selev["splitting_1b0j/1b0j_midMass"] = comb[
            (comb.mass_AB > 75.0) & (comb.mass_AB < 200.0)
        ]
        selev["splitting_1b0j/1b0j_highMass"] = comb[comb.mass_AB > 200.0]

        # DR anomaly check only if the collection exists and non-empty
        if ("splitting_1b0j/1b0j" in selev.fields) and ak.any(
            ak.num(selev["splitting_1b0j/1b0j"]) > 0
        ):
            dr_partA = selev["splitting_1b0j/1b0j"].delta_r(
                selev["splitting_1b0j/1b0j"].part_A
            )
            dr_partB = selev["splitting_1b0j/1b0j"].delta_r(
                selev["splitting_1b0j/1b0j"].part_B
            )
            bad_match_A = ak.any(dr_partA > 1.0, axis=1)
            bad_match_B = ak.any(dr_partB > 1.0, axis=1)
            bad_match_flag = bad_match_A | bad_match_B
            if ak.sum(bad_match_flag) > 0:
                print(
                    f"Found {ak.sum(bad_match_flag)} bad matches in {len(selev['splitting_1b0j/1b0j'])} events"
                )

        # -------- PF assignment and movement (declustered mapping) --------
        def _wrap_phi(phi):
            return (phi + np.pi) % (2 * np.pi) - np.pi

        def _delta_phi(a, b):
            return _wrap_phi(a - b)

        # Assign PFs to original i0/i1 by ΔR
        sj0 = fatjets.i0
        sj1 = fatjets.i1
        pf = fatjets.PFCands

        dr2_0 = (pf.eta - sj0.eta) ** 2 + _wrap_phi(pf.phi - sj0.phi) ** 2
        dr2_1 = (pf.eta - sj1.eta) ** 2 + _wrap_phi(pf.phi - sj1.phi) ** 2
        choice = ak.where(dr2_0 <= dr2_1, 0, 1)
        pf_s0 = pf[choice == 0]
        pf_s1 = pf[choice == 1]

        fatjets = ak.with_field(
            fatjets, ak.zip({"subjet0": pf_s0, "subjet1": pf_s1}, depth_limit=1), "PFCands_by_subjet"
        )
        fatjets = ak.with_field(fatjets, fatjets.PFCands_by_subjet.subjet0, "PFCands_subjet0")
        fatjets = ak.with_field(fatjets, fatjets.PFCands_by_subjet.subjet1, "PFCands_subjet1")


        def movePFCands_ratio(pf_in, old_fj, new_fj):
            dEta = pf_in.eta - old_fj.eta
            dPhi = _delta_phi(pf_in.phi, old_fj.phi)
            pf_pt_b, old_pt_b, new_pt_b, dEta_b, dPhi_b, new_eta_b, new_phi_b = ak.broadcast_arrays(
                pf_in.pt, old_fj.pt, new_fj.pt, dEta, dPhi, new_fj.eta, new_fj.phi
            )
            rPt = ak.where(old_pt_b > 0, pf_pt_b / old_pt_b, 0.0)
            new_pt = rPt * new_pt_b
            new_eta = new_eta_b + dEta_b
            new_phi = _wrap_phi(new_phi_b + dPhi_b)
            out = ak.with_field(pf_in, new_pt, "pt")
            out = ak.with_field(out, new_eta, "eta")
            out = ak.with_field(out, new_phi, "phi")
            return out

        # Subjet-level move
        fatjets = ak.with_field(
            fatjets,
            movePFCands_ratio(fatjets.PFCands_subjet0, fatjets.i0, fatjets.i0_new),
            "PFCands_moved_subjet0",
        )
        fatjets = ak.with_field(
            fatjets,
            movePFCands_ratio(fatjets.PFCands_subjet1, fatjets.i1, fatjets.i1_new),
            "PFCands_moved_subjet1",
        )

        # Fatjet-level move
        fatjets = ak.with_field(
            fatjets, movePFCands_ratio(fatjets.PFCands, fatjets.p, fatjets.p_new), "PFCands_moved"
        )


        # -------- PF-cand deltas (moved - original) --------
        # fatjet-level PF collection
        pf    = fatjets.PFCands
        pf_mv = fatjets.PFCands_moved

        fatjets = ak.with_field(fatjets, pf_mv.pt  - pf.pt,  "pf_dpt")
        fatjets = ak.with_field(fatjets, pf_mv.eta - pf.eta, "pf_deta")
        fatjets = ak.with_field(fatjets, _wrap_phi(pf_mv.phi - pf.phi), "pf_dphi")

        # (optional) subjet0 PF collection
        pf0    = fatjets.PFCands_subjet0
        pf0_mv = fatjets.PFCands_moved_subjet0
        fatjets = ak.with_field(fatjets, pf0_mv.pt  - pf0.pt,  "pf0_dpt")
        fatjets = ak.with_field(fatjets, pf0_mv.eta - pf0.eta, "pf0_deta")
        fatjets = ak.with_field(fatjets, _wrap_phi(pf0_mv.phi - pf0.phi), "pf0_dphi")

        # (optional) subjet1 PF collection
        pf1    = fatjets.PFCands_subjet1
        pf1_mv = fatjets.PFCands_moved_subjet1
        fatjets = ak.with_field(fatjets, pf1_mv.pt  - pf1.pt,  "pf1_dpt")
        fatjets = ak.with_field(fatjets, pf1_mv.eta - pf1.eta, "pf1_deta")
        fatjets = ak.with_field(fatjets, _wrap_phi(pf1_mv.phi - pf1.phi), "pf1_dphi")
        #  Printouts to understand the structure of the PFCands
        #
        # print(f"Number of FatJet PFCands: {ak.num(selev.FatJetPFCands)}\n")
        # print(f"   event fields: {selev.fields}\n")
        # print(f"   FatJet PFCands pt 0: {selev.FatJetPFCands.pt[0].tolist()}\n")
        # print(f"   FatJet PFCands pfCandIdx 0: {selev.FatJetPFCands.pFCandsIdx[0].tolist()}\n")
        # print(f"   FatJet PFCands jetIdx 0: {selev.FatJetPFCands.jetIdx[0].tolist()}\n")
        # print(f"FatJet PFCands fields: {selev.FatJetPFCands.fields}\n")
        # print(f"FatJet Fat Jet fields: {selev.FatJet.fields}\n")
        # print(f"FatJet FatJet dir: {dir(selev.FatJet)}\n")
        # print(f"PFCand fields: {selev.PFCands.fields}\n")

        #print(f" nPFCandsR {ak.num(selev.PFCands[result])}\n")
        #print(f" PFCandsR pdgId {selev.PFCands[result].pdgId.tolist()}\n")


        #print(type(selev.selFatJet),"\n")
        #selev.PFCands[PFCandIndex_FatJet0]


         # ---- Mass difference(subjets & fatjet) ----
        fatjets = ak.with_field(fatjets, fatjets.i0_new.mass - fatjets.i0.mass, "d_mass_i0")
        fatjets = ak.with_field(fatjets, fatjets.i1_new.mass - fatjets.i1.mass, "d_mass_i1")
        fatjets = ak.with_field(fatjets, fatjets.p_new.mass - fatjets.p.mass, "d_mass_fatjet")


        #
        #  Add the fatjets to the event
        #
        selev["fatjets"] = fatjets
        #print(f" PFCands for FatJet1 {selev.FatJetPFCands[PFCandFatJet1_mask].pFCandsIdx.tolist()}\n")
        #print(f"   FatJet PFCands jetIdx 0: {selev.FatJetPFCands.jetIdx[0].tolist()}\n")

        #print(f"Number of selected Fat Jets: {ak.num(selev.selFatJet)}")
        #print(f" Any passNFatJets: {ak.any(selev.passNFatJets)}")
        #print(f" Any passHLT: {ak.any(selev.passHLT)}")
        #print(f" FatJet pt: {selev.selFatJet.pt}")
        #
        #print(f" nFatjets: {ak.num(selev.selFatJet.fatjets, axis=2)}")
        #print(f" subjet pt: {selev.selFatJet.pt[0:10]}")


        # -------- Booking histograms --------
        selev["fourTag"] = True
        selev["tag"] = ak.zip({"fourTag": selev.fourTag})
        selev["SR"] = True
        selev["region"] = ak.zip({"SR": selev.SR})
        selev["weight"] = 1.0

        fill = Fill(process=processName, year=year, weight="weight")
        histCuts = ["passNFatJets"]
        hist = Collection(
            process=[processName],
            year=[year],
            tag=["fourTag"],
            region=["SR"],
            **dict((s, ...) for s in histCuts),
        )

        # Inject btagScore field for plotting API
        selev["selFatJet", "btagScore"] = selev.selFatJet.particleNet_XbbVsQCD
        fill += Jet.plot(
            ("fatJets", "Selected Fat Jets"),
            "selFatJet",
            skip=["deepjet_c"],
            bins={"pt": (50, 0, 1000)},
        )
        # Jets
        fill += hist.add( "msoftdrop",  (100, 40, 400, ("selFatJet.msoftdrop",   'Soft Drop Mass')))

        fill += hist.add("d_mass_i0", (100, -50, 50, ("fatjets.d_mass_i0", "subjet0 mass new-old")))
        fill += hist.add("d_mass_i1", (100, -50, 50, ("fatjets.d_mass_i1", "subjet1 mass new-old")))
        fill += hist.add("d_mass_fatjet", (100, -50, 50, ("fatjets.d_mass_fatjet", "fatjet mass new-old")))

        # ---- PF-cand deltas (moved - original) ----
        fill += hist.add("pf_deta", (100, -0.8, 0.8, ("fatjets.pf_deta", r"PF $\Delta\eta$ (moved - original)")))
        fill += hist.add("pf_dphi", (100, -0.8, 0.8, ("fatjets.pf_dphi", r"PF $\Delta\phi$ (moved - original)")))
        fill += hist.add("pf_dpt",  (100, -5.0, 5.0, ("fatjets.pf_dpt",  r"PF $\Delta p_T$ (moved - original)")))
        # ---- (optional) PF-cand deltas within subjet0 ----
        fill += hist.add("pf0_deta", (100, -0.8, 0.8, ("fatjets.pf0_deta", r"PF $\Delta\eta$ in subjet0 (moved - original)")))
        fill += hist.add("pf0_dphi", (100, -0.8, 0.8, ("fatjets.pf0_dphi", r"PF $\Delta\phi$ in subjet0 (moved - original)")))
        fill += hist.add("pf0_dpt",  (100, -5.0, 5.0, ("fatjets.pf0_dpt",  r"PF $\Delta p_T$ in subjet0 (moved - original)")))

        # ---- (optional) PF-cand deltas within subjet1 ----
        fill += hist.add("pf1_deta", (100, -0.8, 0.8, ("fatjets.pf1_deta", r"PF $\Delta\eta$ in subjet1 (moved - original)")))
        fill += hist.add("pf1_dphi", (100, -0.8, 0.8, ("fatjets.pf1_dphi", r"PF $\Delta\phi$ in subjet1 (moved - original)")))
        fill += hist.add("pf1_dpt",  (100, -5.0, 5.0, ("fatjets.pf1_dpt",  r"PF $\Delta p_T$ in subjet1 (moved - original)")))


        class FatJetHists(Template):
            p = LorentzVector.plot_pair(
                ("...", r"Fat Jet"),
                "p",
                skip=["n"],
                bins={"mass": (100, 40, 300), "pt": (60, 250, 1000), "dr": (50, 0, 1.2), "dphi": (50, -1.5, 1.5)},
            )
            p_new = Jet.plot(("...", r"declustered fatjet"), "p_new", skip=["deepjet_c", "n"], bins={"pt": (60, 250, 1000), "mass": (100, 40, 300)})
            i0 = Jet.plot(("...", r"subjet 0"), "i0", skip=["deepjet_c", "n"], bins={"mass": (100, 0, 200), "pt": (50, 100, 1000)})
            i1 = Jet.plot(("...", r"subjet 1"), "i1", skip=["deepjet_c", "n"], bins={"mass": (50, 0, 100), "pt": (50, 100, 1000)})
            i0_new = Jet.plot(("...", r"declustered subjet 0"), "i0_new", skip=["deepjet_c", "n"], bins={"mass": (100, 0, 200), "pt": (50, 100, 1000)})
            i1_new = Jet.plot(("...", r"declustered subjet 1"), "i1_new", skip=["deepjet_c", "n"], bins={"mass": (50, 0, 100), "pt": (50, 100, 1000)})

            pf = PFCand.plot(("...", r"PFCands in selected fat jet"), "PFCands", skip=[], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})
            pf_moved = PFCand.plot(("...", r"Moved PFCands in selected fat jet"), "PFCands_moved", skip=[], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})
            i0_pf = PFCand.plot(("...", r"PFCands in subjet 0"), "PFCands_subjet0", skip=[], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})
            i1_pf = PFCand.plot(("...", r"PFCands in subjet 1"), "PFCands_subjet1", skip=[], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})
            i0_pf_moved = PFCand.plot(("...", r"Moved PFCands in subjet 0"), "PFCands_moved_subjet0", skip=[], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})
            i1_pf_moved = PFCand.plot(("...", r"Moved PFCands in subjet 1"), "PFCands_moved_subjet1", skip=[], bins={"pt": (50, 0, 10), "mass": (50, 0, 0.2)})

        # Book our fatjets object
        fill += FatJetHists(("fatJets", r""), "fatjets")

        # Per-splitting hists (check existence)
        for s_type in cleaned_splitting_name:
            key = f"splitting_{s_type}"
            if key in selev.fields:
                fill += ClusterHistsBoosted((key, f"{s_type} Splitting"), key)
                fill += ClusterHistsDetailedBoosted((f"detail_{key}", f"{s_type} Splitting"), key)

        # By-mass groups
        fill += ClusterHistsBoosted(("splitting_1b0j/1b0j_lowMass", "1b0j/1b0j Splitting (low Mass)"), "splitting_1b0j/1b0j_lowMass")
        fill += ClusterHistsDetailedBoosted(("detail_splitting_1b0j/1b0j_lowMass", "1b0j/1b0j Splitting (low Mass)"), "splitting_1b0j/1b0j_lowMass")

        fill += ClusterHistsBoosted(("splitting_1b0j/1b0j_midMass", "1b0j/1b0j Splitting (mid Mass)"), "splitting_1b0j/1b0j_midMass")
        fill += ClusterHistsDetailedBoosted(("detail_splitting_1b0j/1b0j_midMass", "1b0j/1b0j Splitting (mid Mass)"), "splitting_1b0j/1b0j_midMass")

        fill += ClusterHistsBoosted(("splitting_1b0j/1b0j_highMass", "1b0j/1b0j Splitting (high Mass)"), "splitting_1b0j/1b0j_highMass")
        fill += ClusterHistsDetailedBoosted(("detail_splitting_1b0j/1b0j_highMass", "1b0j/1b0j Splitting (high Mass)"), "splitting_1b0j/1b0j_highMass")

        # -------- Fill & output --------
        fill(selev, hist)


        print("\n\n================ DEBUG PF (EVENT 0, ALL) START ================\n")

        # if no event，skip
        if len(fatjets) == 0:
            print("No events after mask, skip PF debug.")
        else:
            # only events
            fj0 = fatjets[:1]

            print(">> PF in subjet0 (orig -> moved), EVENT 0")
            print("pf0_eta_orig :", ak.to_list(fj0.PFCands_subjet0.eta))
            print("pf0_eta_moved:", ak.to_list(fj0.PFCands_moved_subjet0.eta))
            print("pf0_phi_orig :", ak.to_list(fj0.PFCands_subjet0.phi))
            print("pf0_phi_moved:", ak.to_list(fj0.PFCands_moved_subjet0.phi))
            print()

            print(">> PF in subjet1 (orig -> moved), EVENT 0")
            print("pf1_eta_orig :", ak.to_list(fj0.PFCands_subjet1.eta))
            print("pf1_eta_moved:", ak.to_list(fj0.PFCands_moved_subjet1.eta))
            print("pf1_phi_orig :", ak.to_list(fj0.PFCands_subjet1.phi))
            print("pf1_phi_moved:", ak.to_list(fj0.PFCands_moved_subjet1.phi))
            print()

            print(">> PF at fatjet-level (orig -> moved), EVENT 0")
            print("pf_eta_orig :", ak.to_list(fj0.PFCands.eta))
            print("pf_eta_moved:", ak.to_list(fj0.PFCands_moved.eta))
            print("pf_phi_orig :", ak.to_list(fj0.PFCands.phi))
            print("pf_phi_moved:", ak.to_list(fj0.PFCands_moved.phi))
            print()

        print("\n================= DEBUG PF (EVENT 0, ALL) END =================\n\n")




        processOutput = {}
        self.cutFlow.addOutput(processOutput, event.metadata["dataset"])
        output = hist.output | processOutput
        return output

    def postprocess(self, accumulator):
        return accumulator
