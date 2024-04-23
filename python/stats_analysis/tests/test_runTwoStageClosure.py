import unittest
import sys
import os
sys.path.insert(0, os.getcwd())
from stats_analysis.tests.parser import wrapper
import ROOT
import numpy as np
import yaml

class TestRunTwoStageClosure(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        #
        # make output
        #

        # This is the output from runTwoStageClosure
        #    time python3 runTwoStageClosure.py
        #
        inputFileName = wrapper.args["inputFile"]
        self.inputROOTFile = ROOT.TFile(inputFileName, "READ")

        #  Make these numbers with:
        #  >  python3     stats_analysis/tests/dumpTwoStageInputs.py --input [inputFileName] -o [outputFielName]
        #    (python3 stats_analysis/tests/dumpTwoStageInputs.py --input stats_analysis/hists_closure_3bDvTMix4bDvT_New.root --output stats_analysis/tests/twoStageClosureInputsCounts.yml)
        knownCountFile = wrapper.args["knownCounts"] 
        self.knownCounts = yaml.safe_load(open(knownCountFile, 'r'))
        print(self.knownCounts.keys())



    def test_input_counts(self):


        for k, v  in self.knownCounts.items():
            channel = v["channel"]
            process = v["process"]
            expected_counts  =  v["counts"]


            isMixed = "mix" in v
            if isMixed: 
                mix = v["mix"]
                print(f"checking .. {mix}/{channel}/{process}")

                input_hist = self.inputROOTFile.Get(f"{mix}/{channel}/{process}")

            else:
                print(f"checking.. {channel}/{process}")
                input_hist = self.inputROOTFile.Get(f"{channel}/{process}")


            intput_counts = []
            for ibin in range(input_hist.GetSize()):
                intput_counts.append(input_hist.GetBinContent(ibin))

            np.testing.assert_array_equal(intput_counts, expected_counts)



if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)


