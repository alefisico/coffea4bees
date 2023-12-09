import unittest
from coffea.util import load

class CutFlowTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        inputFile = "hists/histAll.coffea"
        with open(f'{inputFile}', 'rb') as hfile:
            hists = load(hfile)
        
        self.cf4      = hists["cutFlowFourTag"]
        self.cf4_unit = hists["cutFlowFourTagUnitWeight"]
        self.cf3      = hists["cutFlowThreeTag"]
        self.cf3_unit = hists["cutFlowThreeTagUnitWeight"]

        self.counts4 = {}
        self.counts4["data_UL17C"] = {'passJetMult': 498557, 'passPreSel': 26897,   'passDiJetMass': 12237,   'SR': 4424,   'SB': 7813,   'passSvB': 12,   'failSvB': 15156}
        self.counts4['data_UL17D'] = {'passJetMult': 234161, 'passPreSel': 12959.0, 'passDiJetMass': 5904.0,  'SR': 2065.0, 'SB': 3839.0, 'passSvB': 7.0,  'failSvB': 7372.0}
        self.counts4['data_UL17E'] = {'passJetMult': 549026, 'passPreSel': 26873.0, 'passDiJetMass': 12165.0, 'SR': 4308.0, 'SB': 7857.0, 'passSvB': 14.0, 'failSvB': 15622.0}
        self.counts4['data_UL17F'] = {'passJetMult': 710245, 'passPreSel': 27712.0, 'passDiJetMass': 12225.0, 'SR': 4478.0, 'SB': 7747.0, 'passSvB': 14.0, 'failSvB': 15810.0}

        
        self.counts3 = {}
        self.counts3['data_UL17C'] = {'passJetMult': 498557, 'passPreSel': 21919.6, 'passDiJetMass': 10567.7, 'SR': 3784.1, 'SB': 6783.6, 'passSvB': 15.7, 'failSvB': 12086.0}
        self.counts3['data_UL17D'] = {'passJetMult': 234161, 'passPreSel': 10307.4, 'passDiJetMass': 4987.6,  'SR': 1769.5, 'SB': 3218.0, 'passSvB': 5.1,  'failSvB': 5698.5}
        self.counts3['data_UL17E'] = {'passJetMult': 549026, 'passPreSel': 22916.2, 'passDiJetMass': 11071.6, 'SR': 4000.6, 'SB': 7071.0, 'passSvB': 12.2, 'failSvB': 12947.6}
        self.counts3['data_UL17F'] = {'passJetMult': 710245, 'passPreSel': 30096.1, 'passDiJetMass': 14371.0, 'SR': 5225.0, 'SB': 9145.9, 'passSvB': 16.8, 'failSvB': 16819.3}


        self.counts3_unit = {}
        self.counts3_unit['data_UL17C'] = {'passJetMult': 498557, 'passPreSel': 471660, 'passDiJetMass': 160816, 'SR': 60898, 'SB': 99918,  'passSvB': 181, 'failSvB': 288362}
        self.counts3_unit['data_UL17D'] = {'passJetMult': 234161, 'passPreSel': 221202, 'passDiJetMass': 76166,  'SR': 28620, 'SB': 47546,  'passSvB': 65,  'failSvB': 135202}
        self.counts3_unit['data_UL17E'] = {'passJetMult': 549026, 'passPreSel': 522153, 'passDiJetMass': 179149, 'SR': 67871, 'SB': 111278, 'passSvB': 168, 'failSvB': 327319}
        self.counts3_unit['data_UL17F'] = {'passJetMult': 710245, 'passPreSel': 682533, 'passDiJetMass': 226414, 'SR': 86478, 'SB': 139936, 'passSvB': 197, 'failSvB': 425645}

        

        
    def test_counts4(self):
        """
        Test the cutflow for four tag events
        """
        for datasetAndEra in self.counts4.keys():
            with self.subTest(datasetAndEra=datasetAndEra):
                for cut, v in self.counts4[datasetAndEra].items():
                    self.assertEqual(v,self.cf4[datasetAndEra][cut],f'incorrect number of fourTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts3(self):
        """
        Test the cutflow for the weighted three tag events
        """
        for datasetAndEra in self.counts3.keys():
            for cut, v in self.counts3[datasetAndEra].items():
                self.assertEqual(v,round(self.cf3[datasetAndEra][cut],1),f'incorrect number of weighted threeTag counts for cut: {cut}')

    def test_counts3_unitWeight(self):
        """
        Test the cutflow for the unweighted three tag events
        """
        for datasetAndEra in self.counts3_unit.keys():
            for cut, v in self.counts3_unit[datasetAndEra].items():
                self.assertEqual(v,self.cf3_unit[datasetAndEra][cut],f'incorrect number of threeTag counts for cut: {cut}')

                
                

if __name__ == '__main__':
    unittest.main()
