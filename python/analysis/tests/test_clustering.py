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
from analysis.helpers.clustering import kt_clustering
#import vector
#vector.register_awkward()



import fastjet

#def find_jet_pairs_kernel(events_jets, builder):
#    """Search for valid 4-lepton combinations from an array of events * leptons {charge, ...}
#
#    A valid candidate has two pairs of leptons that each have balanced charge
#    Outputs an array of events * candidates {indices 0..3} corresponding to all valid
#    permutations of all valid combinations of unique leptons in each event
#    (omitting permutations of the pairs)
#    """
#    for jets in events_jets:
#        builder.begin_list()
#        njet = len(events_jets)
#        for i0 in range(njet):
#            for i1 in range(i0 + 1, njet):
#                builder.begin_tuple(2)
#                builder.index(0).integer(i0)
#                builder.index(1).integer(i1)
#                builder.end_tuple()
#        builder.end_list()
#
#    return builder
#
#def find_jet(events_jets):
#
#    # if ak.backend(events_jets) == "typetracer":
#    #    raise Exception("typetracer")
#    #    # here we fake the output of find_4lep_kernel since
#    #    # operating on length-zero data returns the wrong layout!
#    #    ak.typetracer.length_zero_if_typetracer(events_jets.btagDeepFlavB) # force touching of the necessary data
#    #    return ak.Array(ak.Array([[(0,0,0)]]).layout.to_typetracer(forget_length=True))
#    return find_jet_pairs_kernel(events_jets, ak.ArrayBuilder()).snapshot()



class topCandRecoTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
#        self.inputFile = wrapper.args["inputFile"]

#        #
#        # Test vectors from picos
#        #   (All four jet events for now)
#        self.input_jet_pt = [[243, 223, 71.8, 67.8], [181, 119, 116, 96.9], [208, 189, 62.4, 52.1],  [118, 64.6, 58.6, 44.1]]


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


        
    
    def test_clustering_4jets(self):

                
        

#        # Example usage
#        particles = [
#            {'pt': 50, 'eta': 0.3, 'phi': 1.2},
#            {'pt': 30, 'eta': 0.1, 'phi': 1.5},
#            {'pt': 20, 'eta': -0.4, 'phi': -1.2},
#            {'pt': 10, 'eta': 0.3, 'phi': 1.1},
#            # Add more particles as needed
#        ]

        #particles = copy(self.input_jets_4)
        
        R = np.pi  # Jet size parameter
        clustered_jets = kt_clustering(self.input_jets_4, R)

        #print(clustered_jets)
        


        #jetdef04 = fastjet.JetDefinition(fastjet.kt_algorithm, 0.4)
        jetdefAll = fastjet.JetDefinition(fastjet.kt_algorithm, R)
        #cluster04 = fastjet.ClusterSequence(self.input_jets_4, jetdef04)
        clusterAll = fastjet.ClusterSequence(self.input_jets_4, jetdefAll)

#        breakpoint()

        for iEvent, jets in enumerate(clustered_jets):
            print(f"Event {iEvent}")
            for i, jet in enumerate(jets):
                #print(f"Jet {i+1}: pt = {jet['pt']:.2f}, eta = {jet['eta']:.2f}, phi = {jet['phi']:.2f}, mass = {jet.mass:.2f}")
                print(f"Jet {i+1}: px = {jet.px:.2f}, py = {jet.py:.2f}, pz = {jet.pz:.2f}, E = {jet.E:.2f}, type = {jet.jet_flavor}")

            for i_fj, jet_fj in enumerate(clusterAll.inclusive_jets()[iEvent]):
                print(f"FJ  {i_fj+1}: px = {jet_fj.px:.2f}, py = {jet_fj.py:.2f}, pz = {jet_fj.pz:.2f}, E = {jet_fj.E:.2f}")
#                print(f"Jet {i_fj+1}: pt = {jet_fj['pt']:.2f}, eta = {jet_fj['eta']:.2f}, phi = {jet_fj['phi']:.2f}, mass = {jet_fj.mass:.2f}")

                
        
        #breakpoint()
        

    
        
if __name__ == '__main__':
    #wrapper.parse_args()
    #unittest.main(argv=sys.argv)
    unittest.main()
