import unittest
import argparse
from coffea.util import load
import yaml
from parser import wrapper
import sys

#
# python3 analysis/tests/cutflow_test.py   --inputFile hists/test.coffea --knownCounts analysis/tests/testCounts.yml
#
class CutFlowTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        
        inputFile = wrapper.args["inputFile"] #"hists/test.coffea"
        with open(f'{inputFile}', 'rb') as hfile:
            hists = load(hfile)
        
        self.cf4      = hists["cutFlowFourTag"]
        self.cf4_unit = hists["cutFlowFourTagUnitWeight"]
        self.cf3      = hists["cutFlowThreeTag"]
        self.cf3_unit = hists["cutFlowThreeTagUnitWeight"]

        #  Make these numbers with:
        #  >  python     analysis/tests/dumpCutFlow.py --input [inputFileName] -o [outputFielName]
        #       (python analysis/tests/dumpCutFlow.py --input hists/histAll.coffea -o analysis/tests/histAllCounts.yml )
        #
        knownCountFile = wrapper.args["knownCounts"] 
        self.knownCounts = yaml.safe_load(open(knownCountFile, 'r'))

        
    def test_counts4(self):
        """
        Test the cutflow for four tag events
        """
        for datasetAndEra in self.knownCounts["counts4"].keys():
            with self.subTest(datasetAndEra=datasetAndEra):
                for cut, v in self.knownCounts["counts4"][datasetAndEra].items():
                    self.assertEqual(v,round(float(self.cf4[datasetAndEra][cut]),2),f'incorrect number of fourTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts3(self):
        """
        Test the cutflow for the weighted three tag events
        """
        for datasetAndEra in self.knownCounts["counts3"].keys():
            for cut, v in self.knownCounts["counts3"][datasetAndEra].items():
                self.assertEqual(v,round(float(self.cf3[datasetAndEra][cut]),2),f'incorrect number of weighted threeTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts3_unitWeight(self):
        """
        Test the cutflow for the unweighted three tag events
        """
        for datasetAndEra in self.knownCounts["counts3_unit"].keys():
            for cut, v in self.knownCounts["counts3_unit"][datasetAndEra].items():
                self.assertEqual(v,round(float(self.cf3_unit[datasetAndEra][cut]),2),f'incorrect number of threeTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts4_unitWeight(self):
        """
        Test the cutflow for the unweighted fourTag events
        """
        for datasetAndEra in self.knownCounts["counts4_unit"].keys():
            for cut, v in self.knownCounts["counts4_unit"][datasetAndEra].items():
                self.assertEqual(v,round(float(self.cf4_unit[datasetAndEra][cut]),2),f'incorrect number of fourTag counts for cut: {cut} of dataset {datasetAndEra}')


                
                

if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)
    #unittest.main()
