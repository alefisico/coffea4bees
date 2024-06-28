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

sys.path.insert(0, os.getcwd())
from analysis.helpers.clustering import kt_clustering, cluster_bs, decluster_combined_jets, compute_decluster_variables, cluster_bs_fast, make_synthetic_event, get_list_of_splitting_types, clean_ISR

#import vector
#vector.register_awkward()
from coffea.nanoevents.methods.vector import ThreeVector
import fastjet




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

        self.input_jet_pt_5  = [[256.330078125, 133.5384521484375, 120.5, 56.39178466796875, 29.992462158203125], [134.58294677734375, 84.63372802734375, 69.49273681640625, 68.5399169921875, 52.375], [235.428466796875, 190.12060546875, 71.2095947265625, 65.34912109375, 60.4375], [205.2244873046875, 107.375, 94.7037353515625, 93.92022705078125, 92.0400390625], [178.065185546875, 120.625, 94.33807373046875, 84.4742431640625, 72.0380859375], [128.86138916015625, 126.4375, 119.819580078125, 60.375732421875, 47.94287109375], [161.7626953125, 74.89837646484375, 71.490478515625, 56.180023193359375, 50.375], [230.293212890625, 136.250244140625, 62.6171875, 58.7060546875, 49.34375], [228.0, 173.1611328125, 96.181640625, 79.3907470703125, 76.552734375], [128.54522705078125, 106.4375, 84.93209838867188, 64.737548828125, 49.700469970703125]]
        self.input_jet_eta_5  = [[0.8936767578125, 0.121337890625, 1.5341796875, -1.989501953125, -0.7362060546875], [0.29376220703125, 0.9923095703125, 1.52783203125, -0.06246185302734375, -1.863525390625], [2.14892578125, -0.574951171875, -1.082763671875, -0.26849365234375, 0.566650390625], [0.25811767578125, -1.43603515625, 1.063720703125, 1.9384765625, 0.408935546875], [0.46978759765625, 1.232666015625, -1.802734375, -1.022705078125, -1.654052734375], [1.38623046875, 0.6514892578125, 1.039794921875, -0.588623046875, 0.49981689453125], [1.182861328125, 0.174346923828125, -0.564208984375, 1.013671875, -1.2578125], [0.45703125, -0.30816650390625, -0.528564453125, 2.0302734375, -0.859375], [-2.0966796875, -0.52490234375, 0.931640625, 0.096832275390625, -0.21209716796875], [0.046417236328125, -0.7169189453125, 1.510986328125, 0.0444183349609375, 0.19891357421875]]
        self.input_jet_phi_5  = [[2.06884765625, 5.5608954429626465, -1.4248046875, 3.7079901695251465, 1.1357421875], [3.7953925132751465, 0.8619384765625, 0.7774658203125, 6.2029852867126465, 3.04150390625], [1.671142578125, 4.9018378257751465, 5.0817694664001465, 5.3899970054626465, 2.7255859375], [5.0261054039001465, 2.37890625, 2.68701171875, 0.7398681640625, 6.0952277183532715], [1.554931640625, -3.13134765625, 5.6200995445251465, 0.2047119140625, 4.4792304039001465], [2.47021484375, 1.626953125, 4.9301581382751465, 6.0857672691345215, 4.9946112632751465], [1.499267578125, 4.5829901695251465, 4.7106757164001465, 5.0483222007751465, -2.0654296875], [4.3429999351501465, 1.481689453125, 0.78515625, 0.6368408203125, 1.269775390625], [-2.8740234375, 1.008544921875, 4.2407050132751465, 5.5678534507751465, 0.562255859375], [1.16943359375, -2.26611328125, 4.9030585289001465, 5.8507513999938965, 1.7421875]]
        self.input_jet_mass_5  = [[18.3929443359375, 16.167137145996094, 20.8125, 9.837577819824219, 7.376430511474609], [18.627456665039062, 14.282325744628906, 10.098735809326172, 10.676605224609375, 12.8828125], [20.226531982421875, 35.16229248046875, 9.027721405029297, 14.4912109375, 9.1015625], [22.6912841796875, 19.15625, 15.039527893066406, 5.409450531005859, 13.238250732421875], [25.096435546875, 14.2734375, 11.000885009765625, 9.951026916503906, 10.315155029296875], [14.715652465820312, 11.21875, 14.8203125, 8.70376968383789, 8.22491455078125], [20.9124755859375, 10.833625793457031, 11.0955810546875, 10.830390930175781, 9.59375], [23.3994140625, 16.117172241210938, 10.51025390625, 10.590438842773438, 7.5078125], [23.140625, 24.85760498046875, 14.499320983886719, 12.903610229492188, 10.6402587890625], [25.775680541992188, 16.75, 12.032047271728516, 13.135284423828125, 6.7713165283203125]]
        self.input_jet_flavor_5  = [['b', 'b', 'j', 'b', 'b'], ['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'b', 'j'], ['b', 'j', 'b', 'b', 'b'], ['b', 'j', 'b', 'b', 'b'], ['b', 'j', 'b', 'b', 'b'], ['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'b', 'j'], ['j', 'b', 'b', 'b', 'b'], ['b', 'j', 'b', 'b', 'b']]

        self.input_jets_5 = ak.zip(
            {
                "pt": self.input_jet_pt_5,
                "eta": self.input_jet_eta_5,
                "phi": self.input_jet_phi_5,
                "mass": self.input_jet_mass_5,
                "jet_flavor": self.input_jet_flavor_5,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )







        self.debug = False


    def test_kt_clustering_4jets(self):

        R = np.pi  # Jet size parameter
        clustered_jets = kt_clustering(self.input_jets_4, R)


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


    def _declustering_test(self, input_jets, debug=False):

        clustered_jets, clustered_splittings = cluster_bs(input_jets, debug=False)
        compute_decluster_variables(clustered_splittings)

        if debug:
            breakpoint()

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
        pA, pB = decluster_combined_jets(clustered_splittings)


        #
        # Check Pts
        #
        pt_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.pt, (pA + pB).pt)]
        if not all(pt_check):
            [print(i) for i in clustered_splittings.pt - (pA + pB).pt]
            [print(i, j) for i, j in zip(clustered_splittings.pt, (pA + pB).pt)]
        self.assertTrue(all(pt_check), "All pt should be the same")

        #
        # Check Eta
        #
        eta_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.eta, (pA + pB).eta)]
        if not all(eta_check):
            [print(i) for i in clustered_splittings.eta - (pA + pB).eta]
            [print(i, j) for i, j in zip(clustered_splittings.eta, (pA + pB).eta)]
        self.assertTrue(all(eta_check), "All eta should be the same")



        #
        # Check Masses
        #
        mass_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.mass, (pA + pB).mass)]
        if not all(mass_check):
            [print(i) for i in clustered_splittings.mass - (pA + pB).mass]
            [print(i, j) for i, j in zip(clustered_splittings.mass, (pA + pB).mass)]
        self.assertTrue(all(mass_check), "All Masses should be the same")

        #
        # Check Phis
        #
        phi_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.phi, (pA + pB).phi)]
        if not all(phi_check):
            [print(i) for i in clustered_splittings.phi - (pA + pB).phi]
            [print(i, j) for i, j in zip(clustered_splittings.phi, (pA + pB).phi)]
        self.assertTrue(all(phi_check), "All phis should be the same")


    def test_declustering_4jets(self):
        self._declustering_test(self.input_jets_4)

    def test_declustering_5jets(self):
        self._declustering_test(self.input_jets_5, debug=False)


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


        #
        # Check phis
        #
        phi_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.phi, clustered_splittings_fast.phi)]
        if not all(phi_check):
            print("deltas")
            [print(i) for i in clustered_splittings.phi - clustered_splittings_fast.phi]
            print("values")
            [print(i, j) for i, j in zip(clustered_splittings.phi, clustered_splittings_fast.phi)]
        self.assertTrue(all(phi_check), "All phis should be the same")


    def _synthetic_datasets_gbb_only_test(self, input_jets, n_jets_expected, debug=False):

        clustered_jets, _clustered_splittings = cluster_bs(input_jets, debug=False)

        #
        #  Decluster the splitting that are 0b + >1 bs
        #
        if debug: 
            print("Jet flavour Before ISR cleaning")
            print(clustered_jets.jet_flavor)
        clustered_jets = clean_ISR(clustered_jets, _clustered_splittings)
        if debug: 
            print("Jet flavour after ISR cleaning")
            print(clustered_jets.jet_flavor)

        #
        # Declustering
        #

        #
        #  Read in the pdfs
        #
        #  Make with ../.ci-workflows/synthetic-dataset-plot-job.sh
        # input_pdf_file_name = "analysis/plots_synthetic_datasets/clustering_pdfs.yml"
        input_pdf_file_name = "jet_clustering/jet-splitting-PDFs-0jet-00-01-00_5j/clustering_pdfs_vs_pT.yml"
        #input_pdf_file_name = "jet_clustering/clustering_PDFs/clustering_pdfs_vs_pT.yml"
        with open(input_pdf_file_name, 'r') as input_file:
            input_pdfs = yaml.safe_load(input_file)

        declustered_jets = make_synthetic_event(clustered_jets, input_pdfs)
        #pA = declustered_jets[:,0:2]
        #pB = declustered_jets[:,2:]

        #
        # Sanity checks
        #

        match_n_jets = ak.num(declustered_jets) == n_jets_expected
        if not all(match_n_jets):
            print("ERROR number of declustered_jets")
            print(f"ak.num(declustered_jets)        {ak.num(declustered_jets)}")
            print("Only after gbb declustering")
            print(f"clustered_jets.jet_flavor     {clustered_jets.jet_flavor}")
            print(f"clustered_jets.pt             {clustered_jets.pt}")
            #print(f"pA.pt                         {pA.pt}")
            #print(f"pB.pt                         {pB.pt}")

            print(f"declustered_jets.pt             {declustered_jets.pt}")
            print(f"ak.num(declustered_jets)        {ak.num(declustered_jets)}")
            print(f"clustered_jets.phi             {clustered_jets.phi}")

            #
            #  Checkphi
            #
            #print(f"input phi {clustered_jets.phi[1]}")
            #print(f"Reco phi {(pA + pB).phi[1]}")

        self.assertTrue(all(match_n_jets), f"Should always get {n_jets_expected} jets")


    def test_synthetic_datasets_gbb_only_4jets(self):

        self._synthetic_datasets_gbb_only_test(self.input_jets_4, n_jets_expected = 4)


    def test_synthetic_datasets_gbb_only_5jets(self):

        self._synthetic_datasets_gbb_only_test(self.input_jets_5, n_jets_expected = 5, debug=True)


if __name__ == '__main__':
    # wrapper.parse_args()
    # unittest.main(argv=sys.argv)
    unittest.main()
