import unittest
from coffea.util import load

class CutFlowTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        inputFile = "hists/test.coffea"
        with open(f'{inputFile}', 'rb') as hfile:
            hists = load(hfile)
        
        self.cf4      = hists["cutFlowFourTag"]
        self.cf4_unit = hists["cutFlowFourTagUnitWeight"]
        self.cf3      = hists["cutFlowThreeTag"]
        self.cf3_unit = hists["cutFlowThreeTagUnitWeight"]

        #  Make these numbers with:
        #  >  python     analysis/tests/dumpCutFlow.py  -i hists/test.coffea  
        #
        self.counts4 = {}
        self.counts4['data_UL18D'] = {'passJetMult' : 1000.0, 'passPreSel' : 25.0, 'passDiJetMass' : 14.0, 'SR' : 5.0, 'SB' : 9.0, 'passSvB' : 0.0, 'failSvB' : 16.0, }
        self.counts4['data_UL18C'] = {'passJetMult' : 1002.0, 'passPreSel' : 37.0, 'passDiJetMass' : 15.0, 'SR' : 2.0, 'SB' : 13.0, 'passSvB' : 0.0, 'failSvB' : 25.0, }
        self.counts4['data_UL18B'] = {'passJetMult' : 1000.0, 'passPreSel' : 47.0, 'passDiJetMass' : 19.0, 'SR' : 4.0, 'SB' : 15.0, 'passSvB' : 0.0, 'failSvB' : 30.0, }
        self.counts4['data_UL18A'] = {'passJetMult' : 1001.0, 'passPreSel' : 45.0, 'passDiJetMass' : 16.0, 'SR' : 8.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 27.0, }
        self.counts4['ZH4b_UL18'] = {'passJetMult' : 0.22, 'passPreSel' : 0.05, 'passDiJetMass' : 0.05, 'SR' : 0.04, 'SB' : 0.01, 'passSvB' : 0.01, 'failSvB' : 0.0, }
        self.counts4['TTToSemiLeptonic_UL18'] = {'passJetMult' : 11.86, 'passPreSel' : 0.35, 'passDiJetMass' : 0.14, 'SR' : 0.0, 'SB' : 0.13, 'passSvB' : 0.0, 'failSvB' : 0.21, }
        self.counts4['TTToHadronic_UL18'] = {'passJetMult' : 18.38, 'passPreSel' : 0.36, 'passDiJetMass' : 0.18, 'SR' : 0.13, 'SB' : 0.04, 'passSvB' : 0.0, 'failSvB' : 0.15, }
        self.counts4['TTTo2L2Nu_UL18'] = {'passJetMult' : 9.47, 'passPreSel' : 0.52, 'passDiJetMass' : 0.15, 'SR' : 0.04, 'SB' : 0.11, 'passSvB' : 0.02, 'failSvB' : 0.37, }
        self.counts4['HH4b_UL18'] = {'passJetMult' : 0.65, 'passPreSel' : 0.17, 'passDiJetMass' : 0.15, 'SR' : 0.13, 'SB' : 0.02, 'passSvB' : 0.02, 'failSvB' : 0.01, }
        
        
        
        self.counts3 = {}
        self.counts3['data_UL18D'] = {'passJetMult' : 1000.0, 'passPreSel' : 32.64, 'passDiJetMass' : 14.02, 'SR' : 5.29, 'SB' : 8.73, 'passSvB' : 0.03, 'failSvB' : 20.14, }
        self.counts3['data_UL18C'] = {'passJetMult' : 1002.0, 'passPreSel' : 34.1, 'passDiJetMass' : 15.13, 'SR' : 5.81, 'SB' : 9.32, 'passSvB' : 0.0, 'failSvB' : 19.4, }
        self.counts3['data_UL18B'] = {'passJetMult' : 1000.0, 'passPreSel' : 32.08, 'passDiJetMass' : 13.62, 'SR' : 5.82, 'SB' : 7.8, 'passSvB' : 0.0, 'failSvB' : 18.04, }
        self.counts3['data_UL18A'] = {'passJetMult' : 1001.0, 'passPreSel' : 33.38, 'passDiJetMass' : 15.77, 'SR' : 6.24, 'SB' : 9.53, 'passSvB' : 0.0, 'failSvB' : 18.59, }
        self.counts3['ZH4b_UL18'] = {'passJetMult' : 0.22, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['TTToSemiLeptonic_UL18'] = {'passJetMult' : 11.86, 'passPreSel' : 0.37, 'passDiJetMass' : 0.21, 'SR' : 0.1, 'SB' : 0.12, 'passSvB' : 0.0, 'failSvB' : 0.14, }
        self.counts3['TTToHadronic_UL18'] = {'passJetMult' : 18.38, 'passPreSel' : 0.7, 'passDiJetMass' : 0.48, 'SR' : 0.25, 'SB' : 0.23, 'passSvB' : 0.0, 'failSvB' : 0.21, }
        self.counts3['TTTo2L2Nu_UL18'] = {'passJetMult' : 9.47, 'passPreSel' : 0.38, 'passDiJetMass' : 0.2, 'SR' : 0.07, 'SB' : 0.14, 'passSvB' : 0.0, 'failSvB' : 0.19, }
        self.counts3['HH4b_UL18'] = {'passJetMult' : 0.65, 'passPreSel' : 0.02, 'passDiJetMass' : 0.01, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        
        
        
        self.counts4_unit = {}
        self.counts4_unit['data_UL18D'] = {'passJetMult' : 1000.0, 'passPreSel' : 25.0, 'passDiJetMass' : 14.0, 'SR' : 5.0, 'SB' : 9.0, 'passSvB' : 0.0, 'failSvB' : 16.0, }
        self.counts4_unit['data_UL18C'] = {'passJetMult' : 1002.0, 'passPreSel' : 37.0, 'passDiJetMass' : 15.0, 'SR' : 2.0, 'SB' : 13.0, 'passSvB' : 0.0, 'failSvB' : 25.0, }
        self.counts4_unit['data_UL18B'] = {'passJetMult' : 1000.0, 'passPreSel' : 47.0, 'passDiJetMass' : 19.0, 'SR' : 4.0, 'SB' : 15.0, 'passSvB' : 0.0, 'failSvB' : 30.0, }
        self.counts4_unit['data_UL18A'] = {'passJetMult' : 1001.0, 'passPreSel' : 45.0, 'passDiJetMass' : 16.0, 'SR' : 8.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 27.0, }
        self.counts4_unit['ZH4b_UL18'] = {'passJetMult' : 951.0, 'passPreSel' : 172.0, 'passDiJetMass' : 163.0, 'SR' : 128.0, 'SB' : 35.0, 'passSvB' : 13.0, 'failSvB' : 7.0, }
        self.counts4_unit['TTToSemiLeptonic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 13.0, 'passDiJetMass' : 7.0, 'SR' : 1.0, 'SB' : 6.0, 'passSvB' : 0.0, 'failSvB' : 7.0, }
        self.counts4_unit['TTToHadronic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 10.0, 'passDiJetMass' : 5.0, 'SR' : 4.0, 'SB' : 1.0, 'passSvB' : 0.0, 'failSvB' : 4.0, }
        self.counts4_unit['TTTo2L2Nu_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 24.0, 'passDiJetMass' : 8.0, 'SR' : 2.0, 'SB' : 6.0, 'passSvB' : 1.0, 'failSvB' : 17.0, }
        self.counts4_unit['HH4b_UL18'] = {'passJetMult' : 978.0, 'passPreSel' : 214.0, 'passDiJetMass' : 196.0, 'SR' : 173.0, 'SB' : 23.0, 'passSvB' : 21.0, 'failSvB' : 8.0, }
        
        
        
        self.counts3_unit = {}
        self.counts3_unit['data_UL18D'] = {'passJetMult' : 1000.0, 'passPreSel' : 975.0, 'passDiJetMass' : 291.0, 'SR' : 113.0, 'SB' : 178.0, 'passSvB' : 1.0, 'failSvB' : 644.0, }
        self.counts3_unit['data_UL18C'] = {'passJetMult' : 1002.0, 'passPreSel' : 965.0, 'passDiJetMass' : 296.0, 'SR' : 116.0, 'SB' : 180.0, 'passSvB' : 0.0, 'failSvB' : 613.0, }
        self.counts3_unit['data_UL18B'] = {'passJetMult' : 1000.0, 'passPreSel' : 953.0, 'passDiJetMass' : 297.0, 'SR' : 128.0, 'SB' : 169.0, 'passSvB' : 0.0, 'failSvB' : 608.0, }
        self.counts3_unit['data_UL18A'] = {'passJetMult' : 1001.0, 'passPreSel' : 956.0, 'passDiJetMass' : 309.0, 'SR' : 131.0, 'SB' : 178.0, 'passSvB' : 0.0, 'failSvB' : 605.0, }
        self.counts3_unit['ZH4b_UL18'] = {'passJetMult' : 951.0, 'passPreSel' : 777.0, 'passDiJetMass' : 587.0, 'SR' : 377.0, 'SB' : 210.0, 'passSvB' : 8.0, 'failSvB' : 139.0, }
        self.counts3_unit['TTToSemiLeptonic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 988.0, 'passDiJetMass' : 648.0, 'SR' : 332.0, 'SB' : 316.0, 'passSvB' : 3.0, 'failSvB' : 358.0, }
        self.counts3_unit['TTToHadronic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 991.0, 'passDiJetMass' : 738.0, 'SR' : 419.0, 'SB' : 319.0, 'passSvB' : 2.0, 'failSvB' : 268.0, }
        self.counts3_unit['TTTo2L2Nu_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 977.0, 'passDiJetMass' : 504.0, 'SR' : 240.0, 'SB' : 264.0, 'passSvB' : 4.0, 'failSvB' : 496.0, }
        self.counts3_unit['HH4b_UL18'] = {'passJetMult' : 978.0, 'passPreSel' : 760.0, 'passDiJetMass' : 432.0, 'SR' : 307.0, 'SB' : 125.0, 'passSvB' : 26.0, 'failSvB' : 222.0, }

        

        
    def test_counts4(self):
        """
        Test the cutflow for four tag events
        """
        for datasetAndEra in self.counts4.keys():
            with self.subTest(datasetAndEra=datasetAndEra):
                for cut, v in self.counts4[datasetAndEra].items():
                    self.assertEqual(v,round(float(self.cf4[datasetAndEra][cut]),2),f'incorrect number of fourTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts3(self):
        """
        Test the cutflow for the weighted three tag events
        """
        for datasetAndEra in self.counts3.keys():
            for cut, v in self.counts3[datasetAndEra].items():
                self.assertEqual(v,round(float(self.cf3[datasetAndEra][cut]),2),f'incorrect number of weighted threeTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts3_unitWeight(self):
        """
        Test the cutflow for the unweighted three tag events
        """
        for datasetAndEra in self.counts3_unit.keys():
            for cut, v in self.counts3_unit[datasetAndEra].items():
                self.assertEqual(v,round(float(self.cf3_unit[datasetAndEra][cut]),2),f'incorrect number of threeTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts4_unitWeight(self):
        """
        Test the cutflow for the unweighted fourTag events
        """
        for datasetAndEra in self.counts4_unit.keys():
            for cut, v in self.counts4_unit[datasetAndEra].items():
                self.assertEqual(v,round(float(self.cf4_unit[datasetAndEra][cut]),2),f'incorrect number of fourTag counts for cut: {cut} of dataset {datasetAndEra}')


                
                

if __name__ == '__main__':
    unittest.main()
