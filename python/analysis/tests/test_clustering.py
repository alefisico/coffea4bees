import unittest
#import argparse
from coffea.util import load
import yaml
#from parser import wrapper
import sys

import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import time
from copy import copy
import os

try:
    sys.path.insert(0, os.getcwd())
    from analysis.helpers.clustering import kt_clustering, cluster_bs, decluster_combined_jets, compute_decluster_variables, cluster_bs_fast
except:
    sys.path.insert(0, os.getcwd()+"/../..")
    print(sys.path)
    from analysis.helpers.clustering import kt_clustering, cluster_bs, decluster_combined_jets, compute_decluster_variables, cluster_bs_fast

#import vector
#vector.register_awkward()
from coffea.nanoevents.methods.vector import ThreeVector
import fastjet


def rotateZ(particles, angle):
    sinT = np.sin(angle)
    cosT = np.cos(angle)
    x_rotated = cosT * particles.x - sinT * particles.y
    y_rotated = sinT * particles.x + cosT * particles.y

    return ak.zip(
        {
            "x": x_rotated,
            "y": y_rotated,
            "z": particles.z,
            "t": particles.t,
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )


class clusteringTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
#        self.inputFile = wrapper.args["inputFile"]

        #
        # From 4jet events
        #   (from analysis.helpers.topCandReconstruction import dumpTopCandidateTestVectors
        #
        self.input_jet_pt_4  = [[150.25, 127.8125, 60.875, 46.34375], [169.5, 105.3125, 82.375, 63.0], [174.625, 116.5625, 98.875, 58.96875], [297.75, 152.125, 108.3125, 49.875], [232.0, 176.5, 79.4375, 61.84375], [113.5625, 60.78125, 58.84375, 54.40625], [154.875, 137.75, 102.4375, 64.125], [229.75, 156.875, 109.75, 53.21875], [117.125, 94.75, 57.0, 56.1875], [149.875, 116.8125, 97.75, 42.5]]
        self.input_jet_eta_4 = [[0.708740234375, 2.24365234375, -0.37176513671875, -0.5455322265625], [-0.632080078125, 0.4456787109375, -0.27459716796875, -0.04097747802734375], [-2.03564453125, -0.10174560546875, -0.553466796875, -1.352783203125], [-0.26214599609375, -0.5142822265625, 0.6898193359375, 0.8153076171875], [1.438232421875, -0.66015625, 1.7353515625, 2.0283203125], [0.5064697265625, -1.33837890625, 1.122314453125, 0.31939697265625], [1.6240234375, 0.073394775390625, -2.119140625, -2.2568359375], [2.0361328125, 1.675048828125, 0.95849609375, 0.54150390625], [0.14300537109375, 0.9183349609375, -1.08251953125, 1.379150390625], [0.500244140625, -0.0623931884765625, -0.701416015625, -0.88671875]]
        self.input_jet_phi_4 = [[2.4931640625, -0.48309326171875, 2.66259765625, -1.79443359375], [-0.2913818359375, 2.51220703125, -2.73876953125, 0.58349609375], [-2.220703125, 0.6153564453125, 1.251708984375, -1.930908203125], [-1.36962890625, 1.342041015625, 1.99609375, 2.5849609375], [-0.1124420166015625, 2.6875, -2.44775390625, 0.168304443359375], [2.546875, 1.327392578125, -0.794189453125, -0.979248046875], [2.95556640625, 0.7203369140625, -1.276611328125, -0.4969482421875], [1.421630859375, -1.33935546875, -1.302978515625, -3.140625], [-2.45751953125, 0.27557373046875, 1.65087890625, -0.6121826171875], [-3.08984375, -0.14752197265625, 0.2174072265625, -2.95947265625]]
        self.input_jet_mass_4 = [[16.8125, 24.96875, 9.5390625, 6.18359375], [18.859375, 15.296875, 13.5, 7.7421875], [20.5, 16.96875, 11.7265625, 10.7421875], [20.421875, 16.921875, 16.46875, 9.1875], [32.3125, 18.015625, 10.4140625, 13.40625], [14.046875, 9.625, 12.3984375, 8.3515625], [19.3125, 22.875, 13.671875, 12.0234375], [32.15625, 11.8125, 17.25, 11.3828125], [17.0, 14.953125, 9.046875, 11.5], [15.65625, 16.890625, 17.640625, 7.9921875]]

        self.jet_flavor_4 = [["b"] * 4] * len(self.input_jet_pt_4)

        self.input_jets_4 = ak.zip(
            {
                "pt": self.input_jet_pt_4,
                "eta": self.input_jet_eta_4,
                "phi": self.input_jet_phi_4,
                "mass": self.input_jet_mass_4,
                "jet_flavor": self.jet_flavor_4,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        self.debug = False


    def test_kt_clustering_4jets(self):

        R = np.pi  # Jet size parameter
        clustered_jets = kt_clustering(self.input_jets_4, R, remove_mass=False)


        jetdefAll = fastjet.JetDefinition(fastjet.kt_algorithm, R)
        clusterAll = fastjet.ClusterSequence(self.input_jets_4, jetdefAll)

        for iEvent, jets in enumerate(clustered_jets):
            if self.debug: print(f"Event {iEvent}")
            for i, jet in enumerate(jets):

                hasFJMatch = False
                if self.debug: print(f"Jet {i+1}: px = {jet.px:.2f}, py = {jet.py:.2f}, pz = {jet.pz:.2f}, E = {jet.E:.2f}, type = {jet.jet_flavor}")
                for i_fj, jet_fj in enumerate(clusterAll.inclusive_jets()[iEvent]):
                    if np.allclose( (jet.px, jet.py, jet.pz, jet.E),(jet_fj.px, jet_fj.py, jet_fj.pz, jet_fj.E), atol=1e-3 ):
                        if self.debug: print("Has match!")
                        hasFJMatch =True

                self.assertTrue(hasFJMatch, " Not all jets have a fastjet match")

            if self.debug:
                for i_fj, jet_fj in enumerate(clusterAll.inclusive_jets()[iEvent]):
                    print(f"FJ  {i_fj+1}: px = {jet_fj.px:.2f}, py = {jet_fj.py:.2f}, pz = {jet_fj.pz:.2f}, E = {jet_fj.E:.2f}")


    def test_declustering(self):

        clustered_jets, clustered_splittings = cluster_bs(self.input_jets_4, remove_mass=False, debug=False)
        compute_decluster_variables(clustered_splittings)

        if self.debug:
            for iEvent, jets in enumerate(clustered_jets):
                print(f"Event {iEvent}")
                for i, jet in enumerate(jets):
                    print(f"Jet {i+1}: px = {jet.px:.2f}, py = {jet.py:.2f}, pz = {jet.pz:.2f}, E = {jet.E:.2f}, type = {jet.jet_flavor}")
                print("...Splittings")

                for i, splitting in enumerate(clustered_splittings[iEvent]):
                    print(f"Split {i+1}: px = {splitting.px:.2f}, py = {splitting.py:.2f}, pz = {splitting.pz:.2f}, E = {splitting.E:.2f}, type = {splitting.jet_flavor}")
                    print(f"\tPart_A {splitting.part_A}")
                    print(f"\tPart_B {splitting.part_B}")



        #
        # Declustering
        #

        # Eventually will
        #   Lookup thetaA, Z, mA, and mB
        #   radom draw phi  (np.random.uniform(-np.pi, np.pi, len()) ? )
        #
        #  For now use known inputs
        #   (should get exact jets back!)
        clustered_splittings["decluster_mask"] = True
        declustered_jets = decluster_combined_jets(clustered_splittings)


        #
        # Sanity checks
        #
        pA = declustered_jets[:,0:2]
        pB = declustered_jets[:,2:]

        #
        # Check Masses
        #
        mass_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.mass, (pA + pB).mass)]
        if not all(mass_check):
            [print(i) for i in clustered_splittings.mass - (pA + pB).mass]
            [print(i, j) for i, j in zip(clustered_splittings.mass, (pA + pB).mass)]
        self.assertTrue(all(mass_check), "All Masses should be the same")

        #
        # Check Pts
        #
        pt_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.pt, (pA + pB).pt)]
        if not all(pt_check):
            [print(i) for i in clustered_splittings.pt - (pA + pB).pt]
            [print(i, j) for i, j in zip(clustered_splittings.pt, (pA + pB).pt)]
        self.assertTrue(all(pt_check), "All pt should be the same")



    def test_cluster_bs_fast_4jets(self):

        start = time.perf_counter()
        clustered_jets_fast, clustered_splittings_fast = cluster_bs_fast(self.input_jets_4, debug=False)
        end = time.perf_counter()
        elapsed_time_matrix_python = (end - start)
        print(f"\nElapsed time fast Python = {elapsed_time_matrix_python}s")

        start = time.perf_counter()
        clustered_jets, clustered_splittings = cluster_bs(self.input_jets_4, debug=False)
        end = time.perf_counter()
        elapsed_time_loops_python = (end - start)
        print(f"\nElapsed time loops Python = {elapsed_time_loops_python}s")

        #
        # Sanity checks
        #

        #
        # Check Masses
        #
        mass_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.mass, clustered_splittings_fast.mass)]
        if not all(mass_check):
            print("deltas")
            [print(i) for i in clustered_splittings.mass - clustered_splittings_fast.mass]
            print("values")
            [print(i, j) for i, j in zip(clustered_splittings.mass, clustered_splittings_fast.mass)]
        self.assertTrue(all(mass_check), "All Masses should be the same")


        #
        # Check Pts
        #
        pt_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.pt, clustered_splittings_fast.pt)]
        if not all(pt_check):
            print("deltas")
            [print(i) for i in clustered_splittings.pt - clustered_splittings_fast.pt]
            print("values")
            [print(i, j) for i, j in zip(clustered_splittings.pt, clustered_splittings_fast.pt)]
        self.assertTrue(all(pt_check), "All Masses should be the same")



    def test_synthetic_datasets_gbb_only(self):

        clustered_jets, _clustered_splittings = cluster_bs(self.input_jets_4, debug=False)

        g_bb_mask  = clustered_jets.jet_flavor == "g_bb"
        bstar_mask = clustered_jets.jet_flavor == "bstar"

        #
        # 1st replace bstar splittings with their original jets (b, g_bb)
        #
        bstar_mask_splittings = _clustered_splittings.jet_flavor == "bstar"
        bs_from_bstar = _clustered_splittings[bstar_mask_splittings].part_A
        gbbs_from_bstar = _clustered_splittings[bstar_mask_splittings].part_B
        jets_from_bstar = ak.concatenate([bs_from_bstar, gbbs_from_bstar], axis=1)

        clustered_jets_nobStar = clustered_jets[~bstar_mask]
        clustered_jets          = ak.concatenate([clustered_jets_nobStar, jets_from_bstar], axis=1)

        # print(clustered_jets.jet_flavor)
        # print(clustered_jets.pt)
        # print(bs_from_bstar.pt)
        # print(bs_from_bstar.jet_flavor)
        # print(gbbs_from_bstar.pt)
        # print(gbbs_from_bstar.jet_flavor)


        #
        # Declustering
        #
        g_bb_mask  = clustered_jets.jet_flavor == "g_bb"
        bstar_mask = clustered_jets.jet_flavor == "bstar"
        #decluster_mask = g_bb_mask | bstar_mask
        decluster_mask = g_bb_mask  # Just g_bb for now

        #
        #   Lookup thetaA, Z, mA, and mB, decay_phi
        #
        #  Make with ../.ci-workflows/synthetic-dataset-plot-job.sh
        input_pdf_file_name = "analysis/plots_synthetic_datasets/clustering_pdfs.yml"

        n_jets = np.sum(ak.num(clustered_jets))
        n_gbb = np.sum(ak.num(clustered_jets[g_bb_mask]))

        #
        #  Init the random vars
        #
        thetaA    = np.ones(n_jets)
        zA        = np.ones(n_jets)
        decay_phi = np.ones(n_jets)
        mA        = np.ones(n_jets)
        mB        = np.ones(n_jets)

        gbb_indicies = np.where(ak.flatten(g_bb_mask))
        gbb_indicies_tuple = (gbb_indicies[0].to_list())

        bstar_indicies = np.where(ak.flatten(bstar_mask))
        bstar_indicies_tuple = (bstar_indicies[0].to_list())

        #
        #  Read in the pdfs
        #
        with open(input_pdf_file_name, 'r') as input_file:
            input_pdfs = yaml.safe_load(input_file)

        num_samples_gbb   = np.sum(ak.num(clustered_jets[g_bb_mask]))
        num_samples_bstar = np.sum(ak.num(clustered_jets[bstar_mask]))
        varNames = [("gbbs.thetaA", num_samples_gbb), ("gbbs.mA", num_samples_gbb), ("gbbs.mB", num_samples_gbb), ("gbbs.zA", num_samples_gbb), ("gbbs.decay_phi", num_samples_gbb),
                    # ("bstars.thetaA", num_samples_gbb), ("gbbs.mA", num_samples_gbb), ("gbbs.mB", num_samples_gbb), ("gbbs.zA", num_samples_gbb), ("gbbs.decay_phi", num_samples_gbb),
                    ]
        samples = {}
        for _v, _num_samples in varNames:
            probs   = np.array(input_pdfs[_v]["probs"], dtype=float)
            centers = np.array(input_pdfs[_v]["bin_centers"], dtype=float)
            samples[_v] = np.random.choice(centers, size=_num_samples, p=probs)


        thetaA   [gbb_indicies_tuple]   = samples["gbbs.thetaA"]
        zA       [gbb_indicies_tuple]   = samples["gbbs.zA"]
        decay_phi[gbb_indicies_tuple]   = samples["gbbs.decay_phi"]
        mA       [gbb_indicies_tuple]   = samples["gbbs.mA"]
        mB       [gbb_indicies_tuple]   = samples["gbbs.mB"]


        # thetaA   [bstar_indicies_tuple]   = samples["bstars.thetaA"]
        # zA       [bstar_indicies_tuple]   = samples["bstars.zA"]
        # decay_phi[bstar_indicies_tuple]   = samples["bstars.decay_phi"]
        # mA       [bstar_indicies_tuple]   = samples["bstars.mA"]
        # mB       [bstar_indicies_tuple]   = samples["bstars.mB"]

        clustered_jets["decluster_mask"] = decluster_mask
        clustered_jets["thetaA"]         = ak.unflatten(thetaA,    ak.num(clustered_jets))
        clustered_jets["zA"]             = ak.unflatten(zA,        ak.num(clustered_jets))
        clustered_jets["decay_phi"]      = ak.unflatten(decay_phi, ak.num(clustered_jets))
        clustered_jets["mA"]             = ak.unflatten(mA,        ak.num(clustered_jets))
        clustered_jets["mB"]             = ak.unflatten(mB,        ak.num(clustered_jets))

        declustered_jets = decluster_combined_jets(clustered_jets)

        pA = declustered_jets[:,0:2]
        pB = declustered_jets[:,2:]


        #
        # Sanity checks
        #
        print("Only after gbb declustering")
        print(f"clustered_jets.decluster_mask {clustered_jets.decluster_mask}")
        print(f"clustered_jets.jet_flavor     {clustered_jets.jet_flavor}")
        print(f"clustered_jets.pt             {clustered_jets.pt}")
        print(f"pA.pt                         {pA.pt}")
        print(f"pB.pt                         {pB.pt}")

        print(f"declustered_jets.pt             {declustered_jets.pt}")
        print(f"ak.num(declustered_jets)        {ak.num(declustered_jets)}")


if __name__ == '__main__':
    #wrapper.parse_args()
    #unittest.main(argv=sys.argv)
    unittest.main()
