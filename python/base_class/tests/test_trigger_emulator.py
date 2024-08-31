import unittest
#import argparse
#import yaml
#from parser import wrapper
#import sys

from ..trigger_emulator.HLTBTagEmulator import HLTBTagEmulator
from ..trigger_emulator.HLTHtEmulator   import HLTHtEmulator
from ..trigger_emulator.HLTJetEmulator  import HLTJetEmulator
from ..trigger_emulator.TrigEmulator    import TrigEmulator

#import numpy as np
#import awkward as ak
#from coffea.nanoevents.methods import vector
#import time
#from copy import copy
#import os

# sys.path.insert(0, os.getcwd())



class trigger_emulator_TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.debug = False


    def test_HLTBTagEmulator(self):
        emulator = HLTBTagEmulator(high_bin_edge=[10, 20, 30], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])
        result = emulator.passJetThreshold(pt=25, bTagRand=0.15, smearFactor=1.0)
        print(result)

    def test_HLTHtEmulator(self):
        emulator = HLTHtEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])
        result = emulator.passHt(ht=150, seedOffset=1.5, smearFactor=1.0)
        print(result)


    def test_HLTJetEmulator(self):
        emulator = HLTJetEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])
        result = emulator.passJet(pt=150, seedOffset=1.5, smearFactor=1.0)
        print(result)



    def test_passTrig(self):
        # Example objects that might represent thresholds, multiplicities, etc.
        ht_thresholds = [HLTHtEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03]),
                         HLTHtEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])]

        jet_thresholds = [HLTJetEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03]),
                          HLTJetEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])]
        jet_multiplicities = [2, 3]

        btag_op_points = [HLTBTagEmulator(high_bin_edge=[10, 20, 30], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03]),
                          HLTBTagEmulator(high_bin_edge=[10, 20, 30], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03]),]
        btag_multiplicities = [1, 2]

        # Instantiate the trigger emulator
        emulator = TrigEmulator(ht_thresholds, jet_thresholds, jet_multiplicities, btag_op_points, btag_multiplicities)

        # Example data
        offline_jet_pts = [100, 150, 200]
        offline_btagged_jet_pts = [120, 180]
        ht = 500
        seedOffset = 1.2

        # Check if the trigger passes
        result = emulator.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, seedOffset)
        print(result)  # Output: True or False based on the checks





if __name__ == '__main__':
    # wrapper.parse_args()
    # unittest.main(argv=sys.argv)
    unittest.main()
