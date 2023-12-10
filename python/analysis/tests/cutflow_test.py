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

        self.counts4['data_UL16_preVFPB'] = {'passJetMult' : 598690, 'passPreSel' : 17900.0, 'passDiJetMass' : 9211.0, 'SR' : 3424.0, 'SB' : 5787.0, 'passSvB' : 15.0, 'failSvB' : 10432.0, }
        self.counts4['data_UL16_preVFPC'] = {'passJetMult' : 256268, 'passPreSel' : 7263.0, 'passDiJetMass' : 3721.0, 'SR' : 1344.0, 'SB' : 2377.0, 'passSvB' : 4.0, 'failSvB' : 4229.0, }
        self.counts4['data_UL16_preVFPD'] = {'passJetMult' : 421098, 'passPreSel' : 11270.0, 'passDiJetMass' : 5909.0, 'SR' : 2115.0, 'SB' : 3794.0, 'passSvB' : 9.0, 'failSvB' : 6595.0, }
        self.counts4['data_UL16_preVFPE'] = {'passJetMult' : 395976, 'passPreSel' : 9779.0, 'passDiJetMass' : 4964.0, 'SR' : 1810.0, 'SB' : 3154.0, 'passSvB' : 2.0, 'failSvB' : 5762.0, }
        self.counts4['data_UL16_postVFPF'] = {'passJetMult' : 311677, 'passPreSel' : 7731.0, 'passDiJetMass' : 3964.0, 'SR' : 1449.0, 'SB' : 2515.0, 'passSvB' : 6.0, 'failSvB' : 4549.0, }
        self.counts4['data_UL16_postVFPG'] = {'passJetMult' : 877903, 'passPreSel' : 22800.0, 'passDiJetMass' : 12090.0, 'SR' : 4439.0, 'SB' : 7651.0, 'passSvB' : 11.0, 'failSvB' : 13699.0, }
        self.counts4['data_UL16_postVFPH'] = {'passJetMult' : 903644, 'passPreSel' : 24327.0, 'passDiJetMass' : 12698.0, 'SR' : 4676.0, 'SB' : 8022.0, 'passSvB' : 16.0, 'failSvB' : 14548.0, }

        self.counts4["data_UL17C"] = {'passJetMult': 498557, 'passPreSel': 26897,   'passDiJetMass': 12237,   'SR': 4424,   'SB': 7813,   'passSvB': 12,   'failSvB': 15156}
        self.counts4['data_UL17D'] = {'passJetMult': 234161, 'passPreSel': 12959.0, 'passDiJetMass': 5904.0,  'SR': 2065.0, 'SB': 3839.0, 'passSvB': 7.0,  'failSvB': 7372.0}
        self.counts4['data_UL17E'] = {'passJetMult': 549026, 'passPreSel': 26873.0, 'passDiJetMass': 12165.0, 'SR': 4308.0, 'SB': 7857.0, 'passSvB': 14.0, 'failSvB': 15622.0}
        self.counts4['data_UL17F'] = {'passJetMult': 710245, 'passPreSel': 27712.0, 'passDiJetMass': 12225.0, 'SR': 4478.0, 'SB': 7747.0, 'passSvB': 14.0, 'failSvB': 15810.0}

        self.counts4['data_UL18A'] = {'passJetMult': 902005,  'passPreSel': 34344.0, 'passDiJetMass': 14465.0, 'SR': 5486.0,  'SB': 8979.0,  'passSvB': 27.0, 'failSvB': 20510.0}
        self.counts4['data_UL18B'] = {'passJetMult': 448667,  'passPreSel': 16933.0, 'passDiJetMass': 7028.0,  'SR': 2648.0,  'SB': 4380.0,  'passSvB': 8.0,  'failSvB': 10048.0}
        self.counts4['data_UL18C'] = {'passJetMult': 418436,  'passPreSel': 15758.0, 'passDiJetMass': 6560.0,  'SR': 2430.0,  'SB': 4130.0,  'passSvB': 9.0,  'failSvB': 9337.0}
        self.counts4['data_UL18D'] = {'passJetMult': 1874829, 'passPreSel': 70684.0, 'passDiJetMass': 29608.0, 'SR': 11081.0, 'SB': 18527.0, 'passSvB': 56.0, 'failSvB': 41737.0}
        
        
        self.counts3 = {}

        self.counts3['data_UL16_preVFPB'] = {'passJetMult' : 598690, 'passPreSel' : 15419.8, 'passDiJetMass' : 8212.1, 'SR' : 2951.2, 'SB' : 5260.9, 'passSvB' : 12.6, 'failSvB' : 9018.7, }
        self.counts3['data_UL16_preVFPC'] = {'passJetMult' : 256268, 'passPreSel' : 6541.5, 'passDiJetMass' : 3501.8, 'SR' : 1270.1, 'SB' : 2231.7, 'passSvB' : 4.0, 'failSvB' : 3854.6, }
        self.counts3['data_UL16_preVFPD'] = {'passJetMult' : 421098, 'passPreSel' : 10871.5, 'passDiJetMass' : 5832.4, 'SR' : 2110.2, 'SB' : 3722.2, 'passSvB' : 8.0, 'failSvB' : 6405.0, }
        self.counts3['data_UL16_preVFPE'] = {'passJetMult' : 395976, 'passPreSel' : 9943.5, 'passDiJetMass' : 5239.4, 'SR' : 1875.2, 'SB' : 3364.2, 'passSvB' : 6.0, 'failSvB' : 5877.9, }
        self.counts3['data_UL16_postVFPF'] = {'passJetMult' : 311677, 'passPreSel' : 7745.4, 'passDiJetMass' : 4123.3, 'SR' : 1497.4, 'SB' : 2625.8, 'passSvB' : 4.6, 'failSvB' : 4600.4, }
        self.counts3['data_UL16_postVFPG'] = {'passJetMult' : 877903, 'passPreSel' : 21786.4, 'passDiJetMass' : 11875.5, 'SR' : 4318.6, 'SB' : 7556.8, 'passSvB' : 11.8, 'failSvB' : 13001.2, }
        self.counts3['data_UL16_postVFPH'] = {'passJetMult' : 903644, 'passPreSel' : 22182.7, 'passDiJetMass' : 12038.8, 'SR' : 4399.5, 'SB' : 7639.3, 'passSvB' : 11.6, 'failSvB' : 13194.5, }

        self.counts3['data_UL17C'] = {'passJetMult': 498557, 'passPreSel': 21919.6, 'passDiJetMass': 10567.7, 'SR': 3784.1, 'SB': 6783.6, 'passSvB': 15.7, 'failSvB': 12086.0}
        self.counts3['data_UL17D'] = {'passJetMult': 234161, 'passPreSel': 10307.4, 'passDiJetMass': 4987.6,  'SR': 1769.5, 'SB': 3218.0, 'passSvB': 5.1,  'failSvB': 5698.5}
        self.counts3['data_UL17E'] = {'passJetMult': 549026, 'passPreSel': 22916.2, 'passDiJetMass': 11071.6, 'SR': 4000.6, 'SB': 7071.0, 'passSvB': 12.2, 'failSvB': 12947.6}
        self.counts3['data_UL17F'] = {'passJetMult': 710245, 'passPreSel': 30096.1, 'passDiJetMass': 14371.0, 'SR': 5225.0, 'SB': 9145.9, 'passSvB': 16.8, 'failSvB': 16819.3}

        self.counts3['data_UL18A'] = {'passJetMult': 902005, 'passPreSel': 30629.0, 'passDiJetMass': 13741.9, 'SR': 5190.2,  'SB': 8551.7,  'passSvB': 20.6, 'failSvB': 17870.5}
        self.counts3['data_UL18B'] = {'passJetMult': 448667, 'passPreSel': 15304.0, 'passDiJetMass': 6769.8,  'SR': 2559.6,  'SB': 4210.2,  'passSvB': 12.3, 'failSvB': 8846.6}
        self.counts3['data_UL18C'] = {'passJetMult': 418436, 'passPreSel': 14210.1, 'passDiJetMass': 6275.6,  'SR': 2365.5,  'SB': 3910.2,  'passSvB': 11.7, 'failSvB': 8213.2}
        self.counts3['data_UL18D'] = {'passJetMult': 1874829,'passPreSel': 64693.0, 'passDiJetMass': 28721.1, 'SR': 10774.5, 'SB': 17946.7, 'passSvB': 50.7, 'failSvB': 37086.3}
        

        self.counts3_unit = {}

        self.counts3_unit['data_UL16_preVFPB'] = {'passJetMult' : 598690, 'passPreSel' : 580790, 'passDiJetMass' : 241400, 'SR' : 91410, 'SB' : 149990, 'passSvB' : 256, 'failSvB' : 369711, }
        self.counts3_unit['data_UL16_preVFPC'] = {'passJetMult' : 256268, 'passPreSel' : 249005, 'passDiJetMass' : 104103, 'SR' : 39637, 'SB' : 64466, 'passSvB' : 86, 'failSvB' : 159763, }
        self.counts3_unit['data_UL16_preVFPD'] = {'passJetMult' : 421098, 'passPreSel' : 409828, 'passDiJetMass' : 172162, 'SR' : 65369, 'SB' : 106793, 'passSvB' : 174, 'failSvB' : 262289, }
        self.counts3_unit['data_UL16_preVFPE'] = {'passJetMult' : 395976, 'passPreSel' : 386197, 'passDiJetMass' : 158243, 'SR' : 59753, 'SB' : 98490, 'passSvB' : 130, 'failSvB' : 248662, }
        self.counts3_unit['data_UL16_postVFPF'] = {'passJetMult' : 311677, 'passPreSel' : 303946, 'passDiJetMass' : 126079, 'SR' : 47844, 'SB' : 78235, 'passSvB' : 114, 'failSvB' : 196811, }
        self.counts3_unit['data_UL16_postVFPG'] = {'passJetMult' : 877903, 'passPreSel' : 855103, 'passDiJetMass' : 368335, 'SR' : 139637, 'SB' : 228698, 'passSvB' : 244, 'failSvB' : 554649, }
        self.counts3_unit['data_UL16_postVFPH'] = {'passJetMult' : 903644, 'passPreSel' : 879317, 'passDiJetMass' : 375509, 'SR' : 142948, 'SB' : 232561, 'passSvB' : 261, 'failSvB' : 569771, }
        
        self.counts3_unit['data_UL17C'] = {'passJetMult': 498557, 'passPreSel': 471660, 'passDiJetMass': 160816, 'SR': 60898, 'SB': 99918,  'passSvB': 181, 'failSvB': 288362}
        self.counts3_unit['data_UL17D'] = {'passJetMult': 234161, 'passPreSel': 221202, 'passDiJetMass': 76166,  'SR': 28620, 'SB': 47546,  'passSvB': 65,  'failSvB': 135202}
        self.counts3_unit['data_UL17E'] = {'passJetMult': 549026, 'passPreSel': 522153, 'passDiJetMass': 179149, 'SR': 67871, 'SB': 111278, 'passSvB': 168, 'failSvB': 327319}
        self.counts3_unit['data_UL17F'] = {'passJetMult': 710245, 'passPreSel': 682533, 'passDiJetMass': 226414, 'SR': 86478, 'SB': 139936, 'passSvB': 197, 'failSvB': 425645}

        self.counts3_unit['data_UL18A'] = {'passJetMult': 902005,  'passPreSel': 867661,  'passDiJetMass': 270191, 'SR': 106478, 'SB': 163713, 'passSvB': 329, 'failSvB': 561110}
        self.counts3_unit['data_UL18B'] = {'passJetMult': 448667,  'passPreSel': 431734,  'passDiJetMass': 131403, 'SR': 52148,  'SB': 79255,  'passSvB': 209, 'failSvB': 277130}
        self.counts3_unit['data_UL18C'] = {'passJetMult': 418436,  'passPreSel': 402678,  'passDiJetMass': 121773, 'SR': 47810,  'SB': 73963,  'passSvB': 167, 'failSvB': 259193}
        self.counts3_unit['data_UL18D'] = {'passJetMult': 1874829, 'passPreSel': 1804145, 'passDiJetMass': 555111, 'SR': 218599, 'SB': 336512, 'passSvB': 802, 'failSvB': 1149231}        
        

        
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
                self.assertEqual(v,round(self.cf3[datasetAndEra][cut],1),f'incorrect number of weighted threeTag counts for cut: {cut} of dataset {datasetAndEra}')

    def test_counts3_unitWeight(self):
        """
        Test the cutflow for the unweighted three tag events
        """
        for datasetAndEra in self.counts3_unit.keys():
            for cut, v in self.counts3_unit[datasetAndEra].items():
                self.assertEqual(v,self.cf3_unit[datasetAndEra][cut],f'incorrect number of threeTag counts for cut: {cut} of dataset {datasetAndEra}')

                
                

if __name__ == '__main__':
    unittest.main()
