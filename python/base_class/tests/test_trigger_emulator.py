import unittest
#import argparse
import yaml
#from parser import wrapper
#import sys

from ..trigger_emulator.HLTBTagEmulator    import HLTBTagEmulator
from ..trigger_emulator.HLTHtEmulator      import HLTHtEmulator
from ..trigger_emulator.HLTJetEmulator     import HLTJetEmulator
from ..trigger_emulator.TrigEmulator       import TrigEmulator
from ..trigger_emulator.TrigEmulatorTool   import TrigEmulatorTool



class trigger_emulator_TestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Example objects that might represent thresholds, multiplicities, etc.

        self.HtEmulator_test   = HLTHtEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])
        self.BTagEmulator_test = HLTBTagEmulator(high_bin_edge=[10, 20, 30], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])
        self.JetEmulator_test  = HLTJetEmulator(high_bin_edge=[100, 200, 300], eff=[0.1, 0.2, 0.3], eff_err=[0.01, 0.02, 0.03])

        ht_thresholds_test = [self.HtEmulator_test, self.HtEmulator_test]

        jet_thresholds_test = [self.JetEmulator_test, self.JetEmulator_test]

        jet_multiplicities_test = [2, 3]

        btag_op_points_test = [self.BTagEmulator_test, self.BTagEmulator_test]

        btag_multiplicities_test = [1, 2]

        # Instantiate the trigger emulator
        self.emulator_test = TrigEmulator(ht_thresholds_test, jet_thresholds_test, jet_multiplicities_test, btag_op_points_test, btag_multiplicities_test)


        self.emulator_test_tool_data_2018 = TrigEmulatorTool("Test", year="2018")
        self.emulator_test_tool_data_2017 = TrigEmulatorTool("Test", year="2017")
        self.emulator_test_tool_data_2016 = TrigEmulatorTool("Test", year="2016")

        self.emulator_test_tool_mc_2018 = TrigEmulatorTool("Test", year="2018", useMCTurnOns=True)
        self.emulator_test_tool_mc_2017 = TrigEmulatorTool("Test", year="2017", useMCTurnOns=True)
        self.emulator_test_tool_mc_2016 = TrigEmulatorTool("Test", year="2016", useMCTurnOns=True)


        self.emulator_test_tool_data_2018_3b = TrigEmulatorTool("Test", year="2018", is3b=True)
        self.emulator_test_tool_data_2017_3b = TrigEmulatorTool("Test", year="2017", is3b=True)
        self.emulator_test_tool_data_2016_3b = TrigEmulatorTool("Test", year="2016", is3b=True)

        self.emulator_test_tool_mc_2018_3b = TrigEmulatorTool("Test", year="2018", useMCTurnOns=True, is3b=True)
        self.emulator_test_tool_mc_2017_3b = TrigEmulatorTool("Test", year="2017", useMCTurnOns=True, is3b=True)
        self.emulator_test_tool_mc_2016_3b = TrigEmulatorTool("Test", year="2016", useMCTurnOns=True, is3b=True)



        self.debug = False


    def test_HLTBTagEmulator(self):
        emulator = self.BTagEmulator_test
        result = emulator.passJetThreshold(pt=25, bTagRand=0.15, smearFactor=1.0)
        print(result)

    def test_HLTHtEmulator(self):
        emulator = self.HtEmulator_test
        result = emulator.passHt(ht=150, seedOffset=1.5, smearFactor=1.0)
        print(result)


    def test_HLTJetEmulator(self):
        emulator = self.JetEmulator_test
        result = emulator.passJet(pt=150, seedOffset=1.5, smearFactor=1.0)
        print(result)



    def test_passTrig(self):

        # Example data
        offline_jet_pts = [100, 150, 200]
        offline_btagged_jet_pts = [120, 180]
        ht = 500
        seedOffset = 1.2

        # Check if the trigger passes
        result = self.emulator_test.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, seedOffset)
        print(result)  # Output: True or False based on the checks


    def test_calcWeight(self):

        # Example data
        offline_jet_pts = [100, 150, 200]
        offline_btagged_jet_pts = [120, 180]
        ht = 500

        # Calculate the weight
        weight = self.emulator_test.calcWeight(offline_jet_pts, offline_btagged_jet_pts, ht)
        print(weight)  # Output: The calculated weight as a float


    def test_yaml(self):
        input_file_name = "base_class/trigger_emulator/data/haddOutput_All_MC2016_11Nov_fittedTurnOns.yaml"
        with open(input_file_name, 'r') as infile:
            data = yaml.safe_load(infile)

        #print(data.keys())


if __name__ == '__main__':
    # wrapper.parse_args()
    # unittest.main(argv=sys.argv)
    unittest.main()
