import unittest
import argparse
from coffea.util import load
import yaml
from parser import wrapper
import sys

import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import time

import os
sys.path.insert(0, os.getcwd())
from analysis.helpers.topCandReconstruction import find_tops0, find_tops_slow, buildTop


class topCandRecoTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.inputFile = wrapper.args["inputFile"]

        #
        # Test vectors from picos
        #   (All four jet events for now)
        self.input_jet_pt = [[243, 223, 71.8, 67.8], [181, 119, 116, 96.9], [208, 189, 62.4, 52.1],  [118, 64.6, 58.6, 44.1]]
        self.input_jet_eta = [[1.51, 0.516, 0.656, 0.99], [2.02, 0.795, 0.747, 1.45], [1.09, -0.219, 1.5, 0.66],  [0.802, 1.48, 1.56, -0.822]]
        self.input_jet_phi = [[-2.88, 0.5, -1.88, 0.118], [-0.834, 1.87, 2.71, 1.8], [2.21, -0.736, -2.53, 2.12],  [-2.84, 0.558, 2.06, -0.639]]
        self.input_jet_mass = [[28.9, 30.8, 9.66, 13.2], [19.7, 22.8, 15.8, 12.2], [25.8, 24.7, 8.77, 7.87],  [19, 13.1, 10, 7.66] ]
        self.input_jet_btagDeepFlavB = [[0.999, 0.907, 0.0116, 0.685], [0.603, 1, 0.997, 0.646], [0.985, 0.975, 0.0113, 0.967], [0.999, 0.00486, 0.764, 1]]
        self.input_jet_bRegCorr = [[1.1, 1.12, 1.07, 1.02], [0.962, 1, 1.18, 0.977], [1.01, 1.01, 1.09, 1.29],  [1.03, 1.04, 1.18, 0.961]]


        self.output_jet_indices = [[(0, 2, 3), (1, 2, 3)],
                                   [(0, 2, 3), (1, 2, 3)],
                                   [(0, 2, 3), (1, 2, 3)],
                                   [(0, 2, 3), (1, 2, 3)],
                               ]

        # From the c++
        #self.output_xbW =  [5.1162261962890625, 3.6441168785095215, 0.7684999108314514, 5.11277961730957]
        # self.output_xW = [3.4038572311401367, 7.014884948730469, 1.7819664478302002, 0.7852322459220886]

        # From the python
        self.output_xbW = [5.096315417599527, 3.6632132901855132, 0.7625406154799873, 5.152339072484923]
        self.output_xW = [3.4008176212106584, 7.01853836034002, 1.8171329974077641, 0.7874135023333089]



        self.input_jets = ak.zip(
            {
                "pt": self.input_jet_pt,
                "eta": self.input_jet_eta,
                "phi": self.input_jet_phi,
                "mass": self.input_jet_mass,
                "btagDeepFlavB": self.input_jet_btagDeepFlavB,
                "bRegCorr": self.input_jet_bRegCorr,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        self.input_jets = self.input_jets[ak.argsort(self.input_jets.btagDeepFlavB, axis=1, ascending=False)]



#    def test_topCand0_time(self):
#
#        start = time.perf_counter()
#        top_cands = find_tops_slow(self.input_jets)
#        end = time.perf_counter()
#        elapsed_time_bare_python = (end - start)
#        print(f"\nElapsed time Bare Python = {elapsed_time_bare_python}s")
#
#        start = time.perf_counter()
#        top_cands = find_tops0(self.input_jets)
#        end = time.perf_counter()
#        elapsed_time_with_compilation = (end - start)
#        print(f"Elapsed time (with compilation) = {elapsed_time_with_compilation}s")
#
#        start = time.perf_counter()
#        top_cands = find_tops0(self.input_jets)
#        end = time.perf_counter()
#        elapsed_time_after_compilation = (end - start)
#        print(f"Elapsed time (after compilation) = {elapsed_time_after_compilation}s")
#
#        self.assertTrue(elapsed_time_after_compilation < elapsed_time_with_compilation,
#                        f"{elapsed_time_after_compilation} is not less than {elapsed_time_bare_python}")
#



    def test_topCand0(self):

        top_cands0 = find_tops0(self.input_jets)
        for i in range(len(top_cands0)):
            self.assertTrue(np.array_equal(top_cands0[i].to_list(), self.output_jet_indices[i]), "Arrays are not equal")



    def test_buildTopCand(self):
        top_cands0 = find_tops0(self.input_jets)

        rec_top_cands0 = buildTop(self.input_jets, top_cands0)
        xW_min = rec_top_cands0[:, 0].xW
        xbW_min = rec_top_cands0[:, 0].xbW
        #print([xW_min[i] for i in range(len(xW_min))])
        #print([xbW_min[i] for i in range(len(xbW_min))])
        #print(f"xW_min is {xW_min} vs {self.output_xW}  diff {xW_min - self.output_xW}   ({(xW_min - self.output_xW)/self.output_xW})")
        #print(f"xbW_min is {xbW_min} vs {self.output_xbW} diff {xbW_min - self.output_xbW} ({(xbW_min - self.output_xbW)/self.output_xbW})")

        self.assertTrue(np.allclose(xW_min.to_list(), self.output_xW, atol=1e-3), "xW Arrays are not close enough")
        self.assertTrue(np.allclose(xbW_min.to_list(), self.output_xbW, atol=1e-3), "xbW Arrays are not close enough")


        
        

if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)
