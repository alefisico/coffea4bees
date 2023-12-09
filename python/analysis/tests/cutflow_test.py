import unittest
from coffea.util import load

class CutFlowTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print("Ran Setup")
        inputFile = "hists/histAll.coffea"
        with open(f'{inputFile}', 'rb') as hfile:
            hists = load(hfile)
        
        self.cf4      = hists["cutFlowFourTag"]
        self.cf4_unit = hists["cutFlowFourTagUnitWeight"]
        self.cf3      = hists["cutFlowThreeTag"]
        self.cf3_unit = hists["cutFlowThreeTagUnitWeight"]

        self.counts4 = {}
        self.counts4["data_UL17C"] = {'passJetMult': 498557, 'passPreSel': 26897, 'passDiJetMass': 12237, 'SR': 4424, 'SB': 7813, 'passSvB': 12.0, 'failSvB': 15156.0}

        self.counts3 = {}
        self.counts3['data_UL17C'] = {'passJetMult': 498557, 'passPreSel': 21919.6, 'passDiJetMass': 10567.7, 'SR': 3784.1, 'SB': 6783.6, 'passSvB': 15.7, 'failSvB': 12086.0}

        self.counts3_unit = {}
        self.counts3_unit['data_UL17C'] = {'passJetMult': 498557, 'passPreSel': 471660, 'passDiJetMass': 160816, 'SR': 60898, 'SB': 99918, 'passSvB': 181, 'failSvB': 288362}


        
    def test_counts4(self):
        """
        Test the cutflow for four tag events
        """
        datasetAndEra = "data_UL17C"
        for cut, v in self.counts4[datasetAndEra].items():
            with self.subTest(cut=cut):
                self.assertEqual(v,self.cf4[datasetAndEra][cut])

    def test_counts3(self):
        """
        Test the cutflow for the weighted three tag events
        """
        datasetAndEra = "data_UL17C"
        for cut, v in self.counts3[datasetAndEra].items():
            with self.subTest(cut=cut):
                self.assertEqual(v,round(self.cf3[datasetAndEra][cut],1))

    def test_counts3_unitWeight(self):
        """
        Test the cutflow for the unweighted three tag events
        """
        datasetAndEra = "data_UL17C"
        for cut, v in self.counts3_unit[datasetAndEra].items():
            with self.subTest(cut=cut):
                self.assertEqual(v,self.cf3_unit[datasetAndEra][cut])

                
                

if __name__ == '__main__':
    unittest.main()
