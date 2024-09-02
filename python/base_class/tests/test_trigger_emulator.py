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
        print(self.emulator_test_tool_data_2018.GetWeightOR([40,40,40,40], [40,40,40,40], 400))
        print(self.emulator_test_tool_data_2018.GetWeightOR([100,100,100,100], [40,40,40,40], 400))

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

        self.sel_jet_pt_17_4b      = [[94.14671325683594, 68.3095474243164, 64.81517028808594, 62.5611686706543, 49.562095642089844], [175.96209716796875, 95.37551879882812, 77.43939208984375, 73.49510955810547, 68.43262481689453], [128.2674560546875, 79.69510650634766, 61.685760498046875, 60.18799591064453, 51.87247085571289], [192.0273895263672, 87.98374938964844, 78.9425048828125, 73.85894775390625, 57.82630157470703, 44.06169509887695], [146.6157684326172, 125.88066864013672, 108.48120880126953, 83.54424285888672, 43.43476486206055, 42.96830749511719], [87.06474304199219, 84.43431854248047, 54.27083969116211, 45.381954193115234], [128.74998474121094, 74.86860656738281, 70.81383514404297, 58.36476135253906], [139.6393585205078, 136.64593505859375, 54.482662200927734, 49.91389083862305], [205.29588317871094, 181.29681396484375, 142.15841674804688, 75.27525329589844, 46.8840217590332], [296.99310302734375, 240.38710021972656, 182.6641387939453, 102.75593566894531, 49.786102294921875], [98.76945495605469, 70.98324584960938, 62.399757385253906, 40.872886657714844], [129.11929321289062, 103.37531280517578, 100.68328094482422, 65.6379623413086], [185.93093872070312, 109.66726684570312, 77.6453628540039, 56.45912551879883, 43.405948638916016], [146.8939208984375, 135.6363067626953, 100.54847717285156, 76.04730224609375], [101.52064514160156, 61.07514190673828, 53.792572021484375, 40.79029083251953, 40.03313446044922]]
        self.can_jet_pt_17_4b      = [[89.18194580078125, 80.5172119140625, 73.2335433959961, 47.674476623535156], [86.0615005493164, 83.4137191772461, 78.30387115478516, 66.02678680419922], [126.57643127441406, 64.87847900390625, 63.36197280883789, 47.33869552612305], [88.65613555908203, 78.54725646972656, 60.81926345825195, 48.794883728027344], [153.63156127929688, 112.93063354492188, 84.84962463378906, 45.19225311279297], [96.75749969482422, 84.10449981689453, 55.96680450439453, 50.1240119934082], [125.79527282714844, 75.81908416748047, 71.71283721923828, 66.17332458496094], [139.434814453125, 134.5775604248047, 56.23845291137695, 48.50031280517578], [212.51332092285156, 177.04766845703125, 88.72776794433594, 52.19510269165039], [310.9146423339844, 252.3594970703125, 182.48574829101562, 113.79417419433594], [96.7921371459961, 70.4286880493164, 63.19194030761719, 41.43169403076172], [136.3065948486328, 103.73130798339844, 94.5419692993164, 82.04745483398438], [124.01825714111328, 71.69305419921875, 64.28842163085938, 44.508052825927734], [146.3201141357422, 136.1661376953125, 100.35209655761719, 74.11641693115234], [108.1631088256836, 65.42913055419922, 57.78498840332031, 46.640167236328125]]
        self.hT_17_4b              = [339.39471435546875, 616.1602783203125, 381.7087707519531, 534.7005615234375, 609.239990234375, 308.5756530761719, 332.79718017578125, 380.68182373046875, 650.910400390625, 872.5863037109375, 273.0253601074219, 398.81585693359375, 505.93988037109375, 508.5438537597656, 297.2117919921875]
        self.trigWeightMC_17_4b    = [1.0, 1.0, 1.600000023841858, 1.125, 1.25, 1.0, 1.3333332538604736, 1.75, 1.0, 1.0, 3.0, 1.4999998807907104, 1.2857142686843872, 1.25, 0.5]
        self.trigWeightData_17_4b  = [0.5, 0.8999999761581421, 0.5, 0.800000011920929, 0.800000011920929, 0.10000000149011612, 0.30000001192092896, 0.4000000059604645, 1.0, 0.8999999761581421, 0.10000000149011612, 0.6000000238418579, 0.699999988079071, 0.800000011920929, 0.20000000298023224]



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

    def test_jets_17(self):

        for iE in range(len(self.sel_jet_pt_17_4b)):
            _sel_jet_pt = self.sel_jet_pt_17_4b[iE]
            _can_jet_pt = self.can_jet_pt_17_4b[iE]
            _hT         = self.hT_17_4b[iE]
            print(self.emulator_test_tool_data_2017.GetWeightOR(_sel_jet_pt, _can_jet_pt, _hT),"vs",self.trigWeightData_17_4b[iE])
            print("\n",self.emulator_test_tool_mc_2017.GetWeightOR(_sel_jet_pt, _can_jet_pt, _hT),  "vs",self.trigWeightMC_17_4b[iE])


if __name__ == '__main__':
    # wrapper.parse_args()
    # unittest.main(argv=sys.argv)
    unittest.main()
