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
        self.counts4['data_UL17F'] = {'passJetMult' : 1001.0, 'passPreSel' : 40.0, 'passDiJetMass' : 20.0, 'SR' : 8.0, 'SB' : 12.0, 'passSvB' : 0.0, 'failSvB' : 20.0, }
        self.counts4['data_UL17E'] = {'passJetMult' : 1001.0, 'passPreSel' : 44.0, 'passDiJetMass' : 19.0, 'SR' : 11.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 29.0, }
        self.counts4['data_UL17D'] = {'passJetMult' : 1001.0, 'passPreSel' : 56.0, 'passDiJetMass' : 30.0, 'SR' : 10.0, 'SB' : 20.0, 'passSvB' : 0.0, 'failSvB' : 27.0, }
        self.counts4['data_UL17C'] = {'passJetMult' : 1000.0, 'passPreSel' : 55.0, 'passDiJetMass' : 23.0, 'SR' : 9.0, 'SB' : 14.0, 'passSvB' : 0.0, 'failSvB' : 28.0, }
        self.counts4['data_UL16_preVFPE'] = {'passJetMult' : 1000.0, 'passPreSel' : 23.0, 'passDiJetMass' : 13.0, 'SR' : 5.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 13.0, }
        self.counts4['data_UL16_preVFPD'] = {'passJetMult' : 1001.0, 'passPreSel' : 28.0, 'passDiJetMass' : 9.0, 'SR' : 3.0, 'SB' : 6.0, 'passSvB' : 0.0, 'failSvB' : 17.0, }
        self.counts4['data_UL16_preVFPC'] = {'passJetMult' : 1002.0, 'passPreSel' : 28.0, 'passDiJetMass' : 17.0, 'SR' : 5.0, 'SB' : 12.0, 'passSvB' : 0.0, 'failSvB' : 17.0, }
        self.counts4['data_UL16_preVFPB'] = {'passJetMult' : 1000.0, 'passPreSel' : 28.0, 'passDiJetMass' : 19.0, 'SR' : 5.0, 'SB' : 14.0, 'passSvB' : 0.0, 'failSvB' : 16.0, }
        self.counts4['data_UL16_postVFPH'] = {'passJetMult' : 1000.0, 'passPreSel' : 33.0, 'passDiJetMass' : 17.0, 'SR' : 7.0, 'SB' : 10.0, 'passSvB' : 0.0, 'failSvB' : 16.0, }
        self.counts4['data_UL16_postVFPG'] = {'passJetMult' : 1000.0, 'passPreSel' : 30.0, 'passDiJetMass' : 14.0, 'SR' : 5.0, 'SB' : 9.0, 'passSvB' : 0.0, 'failSvB' : 17.0, }
        self.counts4['data_UL16_postVFPF'] = {'passJetMult' : 999.0, 'passPreSel' : 19.0, 'passDiJetMass' : 11.0, 'SR' : 3.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 12.0, }
        self.counts4['ZZ4b_UL18'] = {'passJetMult' : 0.09, 'passPreSel' : 0.02, 'passDiJetMass' : 0.01, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts4['ZZ4b_UL17'] = {'passJetMult' : 0.07, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts4['ZZ4b_UL16_preVFP'] = {'passJetMult' : 0.13, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts4['ZZ4b_UL16_postVFP'] = {'passJetMult' : 0.17, 'passPreSel' : 0.02, 'passDiJetMass' : 0.02, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts4['ZH4b_UL18'] = {'passJetMult' : 0.22, 'passPreSel' : 0.05, 'passDiJetMass' : 0.05, 'SR' : 0.04, 'SB' : 0.01, 'passSvB' : 0.01, 'failSvB' : 0.0, }
        self.counts4['ZH4b_UL17'] = {'passJetMult' : 0.15, 'passPreSel' : 0.04, 'passDiJetMass' : 0.03, 'SR' : 0.03, 'SB' : 0.01, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts4['ZH4b_UL16_preVFP'] = {'passJetMult' : 0.29, 'passPreSel' : 0.05, 'passDiJetMass' : 0.04, 'SR' : 0.04, 'SB' : 0.01, 'passSvB' : 0.01, 'failSvB' : 0.0, }
        self.counts4['ZH4b_UL16_postVFP'] = {'passJetMult' : 0.3, 'passPreSel' : 0.05, 'passDiJetMass' : 0.05, 'SR' : 0.04, 'SB' : 0.01, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts4['TTToSemiLeptonic_UL18'] = {'passJetMult' : 11.86, 'passPreSel' : 0.35, 'passDiJetMass' : 0.14, 'SR' : 0.0, 'SB' : 0.13, 'passSvB' : 0.0, 'failSvB' : 0.21, }
        self.counts4['TTToSemiLeptonic_UL17'] = {'passJetMult' : 10.58, 'passPreSel' : 0.2, 'passDiJetMass' : 0.11, 'SR' : 0.02, 'SB' : 0.1, 'passSvB' : 0.0, 'failSvB' : 0.16, }
        self.counts4['TTToSemiLeptonic_UL16_preVFP'] = {'passJetMult' : 32.03, 'passPreSel' : 0.58, 'passDiJetMass' : 0.43, 'SR' : 0.08, 'SB' : 0.35, 'passSvB' : 0.0, 'failSvB' : 0.27, }
        self.counts4['TTToSemiLeptonic_UL16_postVFP'] = {'passJetMult' : 23.31, 'passPreSel' : 0.37, 'passDiJetMass' : 0.08, 'SR' : 0.04, 'SB' : 0.04, 'passSvB' : 0.0, 'failSvB' : 0.16, }
        self.counts4['TTToHadronic_UL18'] = {'passJetMult' : 18.38, 'passPreSel' : 0.36, 'passDiJetMass' : 0.18, 'SR' : 0.13, 'SB' : 0.04, 'passSvB' : 0.0, 'failSvB' : 0.15, }
        self.counts4['TTToHadronic_UL17'] = {'passJetMult' : 16.37, 'passPreSel' : 0.24, 'passDiJetMass' : 0.17, 'SR' : 0.11, 'SB' : 0.06, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts4['TTToHadronic_UL16_preVFP'] = {'passJetMult' : 49.74, 'passPreSel' : 0.53, 'passDiJetMass' : 0.19, 'SR' : 0.13, 'SB' : 0.05, 'passSvB' : 0.0, 'failSvB' : 0.32, }
        self.counts4['TTToHadronic_UL16_postVFP'] = {'passJetMult' : 36.84, 'passPreSel' : 0.31, 'passDiJetMass' : 0.14, 'SR' : 0.1, 'SB' : 0.05, 'passSvB' : 0.0, 'failSvB' : 0.1, }
        self.counts4['TTTo2L2Nu_UL18'] = {'passJetMult' : 9.47, 'passPreSel' : 0.52, 'passDiJetMass' : 0.15, 'SR' : 0.04, 'SB' : 0.11, 'passSvB' : 0.02, 'failSvB' : 0.37, }
        self.counts4['TTTo2L2Nu_UL17'] = {'passJetMult' : 8.85, 'passPreSel' : 0.49, 'passDiJetMass' : 0.2, 'SR' : 0.09, 'SB' : 0.11, 'passSvB' : 0.0, 'failSvB' : 0.21, }
        self.counts4['TTTo2L2Nu_UL16_preVFP'] = {'passJetMult' : 25.36, 'passPreSel' : 0.42, 'passDiJetMass' : 0.25, 'SR' : 0.1, 'SB' : 0.15, 'passSvB' : 0.0, 'failSvB' : 0.22, }
        self.counts4['TTTo2L2Nu_UL16_postVFP'] = {'passJetMult' : 18.29, 'passPreSel' : 0.65, 'passDiJetMass' : 0.2, 'SR' : 0.12, 'SB' : 0.07, 'passSvB' : 0.0, 'failSvB' : 0.45, }
        self.counts4['HH4b_UL18'] = {'passJetMult' : 0.65, 'passPreSel' : 0.17, 'passDiJetMass' : 0.15, 'SR' : 0.13, 'SB' : 0.02, 'passSvB' : 0.02, 'failSvB' : 0.01, }
        self.counts4['HH4b_UL17'] = {'passJetMult' : 0.4, 'passPreSel' : 0.11, 'passDiJetMass' : 0.1, 'SR' : 0.09, 'SB' : 0.01, 'passSvB' : 0.02, 'failSvB' : 0.01, }
        self.counts4['HH4b_UL16_preVFP'] = {'passJetMult' : 0.51, 'passPreSel' : 0.09, 'passDiJetMass' : 0.08, 'SR' : 0.07, 'SB' : 0.01, 'passSvB' : 0.01, 'failSvB' : 0.01, }
        
        
        
        self.counts3 = {}
        self.counts3['data_UL18D'] = {'passJetMult' : 1000.0, 'passPreSel' : 32.64, 'passDiJetMass' : 14.02, 'SR' : 5.29, 'SB' : 8.73, 'passSvB' : 0.03, 'failSvB' : 20.14, }
        self.counts3['data_UL18C'] = {'passJetMult' : 1002.0, 'passPreSel' : 34.1, 'passDiJetMass' : 15.13, 'SR' : 5.81, 'SB' : 9.32, 'passSvB' : 0.0, 'failSvB' : 19.4, }
        self.counts3['data_UL18B'] = {'passJetMult' : 1000.0, 'passPreSel' : 32.08, 'passDiJetMass' : 13.62, 'SR' : 5.82, 'SB' : 7.8, 'passSvB' : 0.0, 'failSvB' : 18.04, }
        self.counts3['data_UL18A'] = {'passJetMult' : 1001.0, 'passPreSel' : 33.38, 'passDiJetMass' : 15.77, 'SR' : 6.24, 'SB' : 9.53, 'passSvB' : 0.0, 'failSvB' : 18.59, }
        self.counts3['data_UL17F'] = {'passJetMult' : 1001.0, 'passPreSel' : 44.66, 'passDiJetMass' : 21.54, 'SR' : 8.2, 'SB' : 13.33, 'passSvB' : 0.16, 'failSvB' : 25.17, }
        self.counts3['data_UL17E'] = {'passJetMult' : 1001.0, 'passPreSel' : 43.04, 'passDiJetMass' : 20.88, 'SR' : 6.96, 'SB' : 13.92, 'passSvB' : 0.08, 'failSvB' : 25.24, }
        self.counts3['data_UL17D'] = {'passJetMult' : 1001.0, 'passPreSel' : 44.61, 'passDiJetMass' : 22.38, 'SR' : 8.81, 'SB' : 13.57, 'passSvB' : 0.0, 'failSvB' : 24.04, }
        self.counts3['data_UL17C'] = {'passJetMult' : 1000.0, 'passPreSel' : 44.18, 'passDiJetMass' : 23.59, 'SR' : 7.9, 'SB' : 15.7, 'passSvB' : 0.06, 'failSvB' : 23.4, }
        self.counts3['data_UL16_preVFPE'] = {'passJetMult' : 1000.0, 'passPreSel' : 25.43, 'passDiJetMass' : 13.39, 'SR' : 5.14, 'SB' : 8.25, 'passSvB' : 0.0, 'failSvB' : 14.52, }
        self.counts3['data_UL16_preVFPD'] = {'passJetMult' : 1001.0, 'passPreSel' : 25.63, 'passDiJetMass' : 13.35, 'SR' : 5.19, 'SB' : 8.16, 'passSvB' : 0.11, 'failSvB' : 15.26, }
        self.counts3['data_UL16_preVFPC'] = {'passJetMult' : 1002.0, 'passPreSel' : 27.27, 'passDiJetMass' : 15.59, 'SR' : 5.22, 'SB' : 10.37, 'passSvB' : 0.01, 'failSvB' : 15.67, }
        self.counts3['data_UL16_preVFPB'] = {'passJetMult' : 1000.0, 'passPreSel' : 25.69, 'passDiJetMass' : 13.31, 'SR' : 4.13, 'SB' : 9.18, 'passSvB' : 0.0, 'failSvB' : 15.18, }
        self.counts3['data_UL16_postVFPH'] = {'passJetMult' : 1000.0, 'passPreSel' : 24.7, 'passDiJetMass' : 13.91, 'SR' : 6.04, 'SB' : 7.87, 'passSvB' : 0.0, 'failSvB' : 14.14, }
        self.counts3['data_UL16_postVFPG'] = {'passJetMult' : 1000.0, 'passPreSel' : 25.96, 'passDiJetMass' : 13.62, 'SR' : 4.71, 'SB' : 8.91, 'passSvB' : 0.0, 'failSvB' : 15.28, }
        self.counts3['data_UL16_postVFPF'] = {'passJetMult' : 999.0, 'passPreSel' : 22.83, 'passDiJetMass' : 12.92, 'SR' : 4.32, 'SB' : 8.6, 'passSvB' : 0.0, 'failSvB' : 13.49, }
        self.counts3['ZZ4b_UL18'] = {'passJetMult' : 0.09, 'passPreSel' : 0.0, 'passDiJetMass' : 0.0, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : -0.0, 'failSvB' : 0.0, }
        self.counts3['ZZ4b_UL17'] = {'passJetMult' : 0.07, 'passPreSel' : 0.0, 'passDiJetMass' : 0.0, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : -0.0, 'failSvB' : 0.0, }
        self.counts3['ZZ4b_UL16_preVFP'] = {'passJetMult' : 0.13, 'passPreSel' : 0.0, 'passDiJetMass' : 0.0, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['ZZ4b_UL16_postVFP'] = {'passJetMult' : 0.17, 'passPreSel' : 0.0, 'passDiJetMass' : 0.0, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['ZH4b_UL18'] = {'passJetMult' : 0.22, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['ZH4b_UL17'] = {'passJetMult' : 0.15, 'passPreSel' : 0.01, 'passDiJetMass' : 0.0, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['ZH4b_UL16_preVFP'] = {'passJetMult' : 0.29, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['ZH4b_UL16_postVFP'] = {'passJetMult' : 0.3, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.0, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['TTToSemiLeptonic_UL18'] = {'passJetMult' : 11.86, 'passPreSel' : 0.37, 'passDiJetMass' : 0.21, 'SR' : 0.1, 'SB' : 0.12, 'passSvB' : 0.0, 'failSvB' : 0.14, }
        self.counts3['TTToSemiLeptonic_UL17'] = {'passJetMult' : 10.58, 'passPreSel' : 0.38, 'passDiJetMass' : 0.24, 'SR' : 0.11, 'SB' : 0.14, 'passSvB' : 0.0, 'failSvB' : 0.13, }
        self.counts3['TTToSemiLeptonic_UL16_preVFP'] = {'passJetMult' : 32.03, 'passPreSel' : 0.82, 'passDiJetMass' : 0.52, 'SR' : 0.23, 'SB' : 0.3, 'passSvB' : 0.0, 'failSvB' : 0.27, }
        self.counts3['TTToSemiLeptonic_UL16_postVFP'] = {'passJetMult' : 23.31, 'passPreSel' : 0.58, 'passDiJetMass' : 0.37, 'SR' : 0.17, 'SB' : 0.2, 'passSvB' : 0.01, 'failSvB' : 0.23, }
        self.counts3['TTToHadronic_UL18'] = {'passJetMult' : 18.38, 'passPreSel' : 0.7, 'passDiJetMass' : 0.48, 'SR' : 0.25, 'SB' : 0.23, 'passSvB' : 0.0, 'failSvB' : 0.21, }
        self.counts3['TTToHadronic_UL17'] = {'passJetMult' : 16.37, 'passPreSel' : 0.69, 'passDiJetMass' : 0.5, 'SR' : 0.25, 'SB' : 0.26, 'passSvB' : 0.0, 'failSvB' : 0.19, }
        self.counts3['TTToHadronic_UL16_preVFP'] = {'passJetMult' : 49.74, 'passPreSel' : 1.33, 'passDiJetMass' : 1.0, 'SR' : 0.46, 'SB' : 0.54, 'passSvB' : 0.0, 'failSvB' : 0.35, }
        self.counts3['TTToHadronic_UL16_postVFP'] = {'passJetMult' : 36.84, 'passPreSel' : 1.05, 'passDiJetMass' : 0.79, 'SR' : 0.39, 'SB' : 0.39, 'passSvB' : 0.0, 'failSvB' : 0.26, }
        self.counts3['TTTo2L2Nu_UL18'] = {'passJetMult' : 9.47, 'passPreSel' : 0.38, 'passDiJetMass' : 0.2, 'SR' : 0.07, 'SB' : 0.14, 'passSvB' : 0.0, 'failSvB' : 0.19, }
        self.counts3['TTTo2L2Nu_UL17'] = {'passJetMult' : 8.85, 'passPreSel' : 0.33, 'passDiJetMass' : 0.18, 'SR' : 0.07, 'SB' : 0.11, 'passSvB' : 0.0, 'failSvB' : 0.13, }
        self.counts3['TTTo2L2Nu_UL16_preVFP'] = {'passJetMult' : 25.36, 'passPreSel' : 0.63, 'passDiJetMass' : 0.33, 'SR' : 0.14, 'SB' : 0.19, 'passSvB' : 0.0, 'failSvB' : 0.26, }
        self.counts3['TTTo2L2Nu_UL16_postVFP'] = {'passJetMult' : 18.29, 'passPreSel' : 0.47, 'passDiJetMass' : 0.26, 'SR' : 0.11, 'SB' : 0.15, 'passSvB' : 0.0, 'failSvB' : 0.22, }
        self.counts3['HH4b_UL18'] = {'passJetMult' : 0.65, 'passPreSel' : 0.02, 'passDiJetMass' : 0.01, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['HH4b_UL17'] = {'passJetMult' : 0.4, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        self.counts3['HH4b_UL16_preVFP'] = {'passJetMult' : 0.51, 'passPreSel' : 0.01, 'passDiJetMass' : 0.01, 'SR' : 0.01, 'SB' : 0.0, 'passSvB' : 0.0, 'failSvB' : 0.0, }
        
        
        
        self.counts4_unit = {}
        self.counts4_unit['data_UL18D'] = {'passJetMult' : 1000.0, 'passPreSel' : 25.0, 'passDiJetMass' : 14.0, 'SR' : 5.0, 'SB' : 9.0, 'passSvB' : 0.0, 'failSvB' : 16.0, }
        self.counts4_unit['data_UL18C'] = {'passJetMult' : 1002.0, 'passPreSel' : 37.0, 'passDiJetMass' : 15.0, 'SR' : 2.0, 'SB' : 13.0, 'passSvB' : 0.0, 'failSvB' : 25.0, }
        self.counts4_unit['data_UL18B'] = {'passJetMult' : 1000.0, 'passPreSel' : 47.0, 'passDiJetMass' : 19.0, 'SR' : 4.0, 'SB' : 15.0, 'passSvB' : 0.0, 'failSvB' : 30.0, }
        self.counts4_unit['data_UL18A'] = {'passJetMult' : 1001.0, 'passPreSel' : 45.0, 'passDiJetMass' : 16.0, 'SR' : 8.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 27.0, }
        self.counts4_unit['data_UL17F'] = {'passJetMult' : 1001.0, 'passPreSel' : 40.0, 'passDiJetMass' : 20.0, 'SR' : 8.0, 'SB' : 12.0, 'passSvB' : 0.0, 'failSvB' : 20.0, }
        self.counts4_unit['data_UL17E'] = {'passJetMult' : 1001.0, 'passPreSel' : 44.0, 'passDiJetMass' : 19.0, 'SR' : 11.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 29.0, }
        self.counts4_unit['data_UL17D'] = {'passJetMult' : 1001.0, 'passPreSel' : 56.0, 'passDiJetMass' : 30.0, 'SR' : 10.0, 'SB' : 20.0, 'passSvB' : 0.0, 'failSvB' : 27.0, }
        self.counts4_unit['data_UL17C'] = {'passJetMult' : 1000.0, 'passPreSel' : 55.0, 'passDiJetMass' : 23.0, 'SR' : 9.0, 'SB' : 14.0, 'passSvB' : 0.0, 'failSvB' : 28.0, }
        self.counts4_unit['data_UL16_preVFPE'] = {'passJetMult' : 1000.0, 'passPreSel' : 23.0, 'passDiJetMass' : 13.0, 'SR' : 5.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 13.0, }
        self.counts4_unit['data_UL16_preVFPD'] = {'passJetMult' : 1001.0, 'passPreSel' : 28.0, 'passDiJetMass' : 9.0, 'SR' : 3.0, 'SB' : 6.0, 'passSvB' : 0.0, 'failSvB' : 17.0, }
        self.counts4_unit['data_UL16_preVFPC'] = {'passJetMult' : 1002.0, 'passPreSel' : 28.0, 'passDiJetMass' : 17.0, 'SR' : 5.0, 'SB' : 12.0, 'passSvB' : 0.0, 'failSvB' : 17.0, }
        self.counts4_unit['data_UL16_preVFPB'] = {'passJetMult' : 1000.0, 'passPreSel' : 28.0, 'passDiJetMass' : 19.0, 'SR' : 5.0, 'SB' : 14.0, 'passSvB' : 0.0, 'failSvB' : 16.0, }
        self.counts4_unit['data_UL16_postVFPH'] = {'passJetMult' : 1000.0, 'passPreSel' : 33.0, 'passDiJetMass' : 17.0, 'SR' : 7.0, 'SB' : 10.0, 'passSvB' : 0.0, 'failSvB' : 16.0, }
        self.counts4_unit['data_UL16_postVFPG'] = {'passJetMult' : 1000.0, 'passPreSel' : 30.0, 'passDiJetMass' : 14.0, 'SR' : 5.0, 'SB' : 9.0, 'passSvB' : 0.0, 'failSvB' : 17.0, }
        self.counts4_unit['data_UL16_postVFPF'] = {'passJetMult' : 999.0, 'passPreSel' : 19.0, 'passDiJetMass' : 11.0, 'SR' : 3.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 12.0, }
        self.counts4_unit['ZZ4b_UL18'] = {'passJetMult' : 946.0, 'passPreSel' : 119.0, 'passDiJetMass' : 108.0, 'SR' : 87.0, 'SB' : 21.0, 'passSvB' : 5.0, 'failSvB' : 8.0, }
        self.counts4_unit['ZZ4b_UL17'] = {'passJetMult' : 951.0, 'passPreSel' : 128.0, 'passDiJetMass' : 110.0, 'SR' : 80.0, 'SB' : 30.0, 'passSvB' : 6.0, 'failSvB' : 16.0, }
        self.counts4_unit['ZZ4b_UL16_preVFP'] = {'passJetMult' : 966.0, 'passPreSel' : 107.0, 'passDiJetMass' : 94.0, 'SR' : 70.0, 'SB' : 24.0, 'passSvB' : 3.0, 'failSvB' : 13.0, }
        self.counts4_unit['ZZ4b_UL16_postVFP'] = {'passJetMult' : 972.0, 'passPreSel' : 102.0, 'passDiJetMass' : 93.0, 'SR' : 68.0, 'SB' : 25.0, 'passSvB' : 4.0, 'failSvB' : 11.0, }
        self.counts4_unit['ZH4b_UL18'] = {'passJetMult' : 951.0, 'passPreSel' : 172.0, 'passDiJetMass' : 163.0, 'SR' : 128.0, 'SB' : 35.0, 'passSvB' : 13.0, 'failSvB' : 7.0, }
        self.counts4_unit['ZH4b_UL17'] = {'passJetMult' : 947.0, 'passPreSel' : 187.0, 'passDiJetMass' : 171.0, 'SR' : 147.0, 'SB' : 24.0, 'passSvB' : 8.0, 'failSvB' : 8.0, }
        self.counts4_unit['ZH4b_UL16_preVFP'] = {'passJetMult' : 955.0, 'passPreSel' : 146.0, 'passDiJetMass' : 136.0, 'SR' : 112.0, 'SB' : 24.0, 'passSvB' : 14.0, 'failSvB' : 12.0, }
        self.counts4_unit['ZH4b_UL16_postVFP'] = {'passJetMult' : 956.0, 'passPreSel' : 169.0, 'passDiJetMass' : 159.0, 'SR' : 128.0, 'SB' : 31.0, 'passSvB' : 9.0, 'failSvB' : 8.0, }
        self.counts4_unit['TTToSemiLeptonic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 13.0, 'passDiJetMass' : 7.0, 'SR' : 1.0, 'SB' : 6.0, 'passSvB' : 0.0, 'failSvB' : 7.0, }
        self.counts4_unit['TTToSemiLeptonic_UL17'] = {'passJetMult' : 1000.0, 'passPreSel' : 9.0, 'passDiJetMass' : 5.0, 'SR' : 1.0, 'SB' : 4.0, 'passSvB' : 0.0, 'failSvB' : 7.0, }
        self.counts4_unit['TTToSemiLeptonic_UL16_preVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 11.0, 'passDiJetMass' : 8.0, 'SR' : 2.0, 'SB' : 6.0, 'passSvB' : 0.0, 'failSvB' : 4.0, }
        self.counts4_unit['TTToSemiLeptonic_UL16_postVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 13.0, 'passDiJetMass' : 5.0, 'SR' : 3.0, 'SB' : 2.0, 'passSvB' : 0.0, 'failSvB' : 7.0, }
        self.counts4_unit['TTToHadronic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 10.0, 'passDiJetMass' : 5.0, 'SR' : 4.0, 'SB' : 1.0, 'passSvB' : 0.0, 'failSvB' : 4.0, }
        self.counts4_unit['TTToHadronic_UL17'] = {'passJetMult' : 1000.0, 'passPreSel' : 7.0, 'passDiJetMass' : 4.0, 'SR' : 2.0, 'SB' : 2.0, 'passSvB' : 0.0, 'failSvB' : 1.0, }
        self.counts4_unit['TTToHadronic_UL16_preVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 7.0, 'passDiJetMass' : 2.0, 'SR' : 1.0, 'SB' : 1.0, 'passSvB' : 0.0, 'failSvB' : 5.0, }
        self.counts4_unit['TTToHadronic_UL16_postVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 8.0, 'passDiJetMass' : 5.0, 'SR' : 3.0, 'SB' : 2.0, 'passSvB' : 0.0, 'failSvB' : 2.0, }
        self.counts4_unit['TTTo2L2Nu_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 24.0, 'passDiJetMass' : 8.0, 'SR' : 2.0, 'SB' : 6.0, 'passSvB' : 1.0, 'failSvB' : 17.0, }
        self.counts4_unit['TTTo2L2Nu_UL17'] = {'passJetMult' : 1000.0, 'passPreSel' : 28.0, 'passDiJetMass' : 14.0, 'SR' : 6.0, 'SB' : 8.0, 'passSvB' : 0.0, 'failSvB' : 12.0, }
        self.counts4_unit['TTTo2L2Nu_UL16_preVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 15.0, 'passDiJetMass' : 8.0, 'SR' : 3.0, 'SB' : 5.0, 'passSvB' : 0.0, 'failSvB' : 8.0, }
        self.counts4_unit['TTTo2L2Nu_UL16_postVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 25.0, 'passDiJetMass' : 8.0, 'SR' : 6.0, 'SB' : 2.0, 'passSvB' : 0.0, 'failSvB' : 18.0, }
        self.counts4_unit['HH4b_UL18'] = {'passJetMult' : 978.0, 'passPreSel' : 214.0, 'passDiJetMass' : 196.0, 'SR' : 173.0, 'SB' : 23.0, 'passSvB' : 21.0, 'failSvB' : 8.0, }
        self.counts4_unit['HH4b_UL17'] = {'passJetMult' : 974.0, 'passPreSel' : 209.0, 'passDiJetMass' : 189.0, 'SR' : 170.0, 'SB' : 19.0, 'passSvB' : 29.0, 'failSvB' : 11.0, }
        self.counts4_unit['HH4b_UL16_preVFP'] = {'passJetMult' : 982.0, 'passPreSel' : 180.0, 'passDiJetMass' : 161.0, 'SR' : 137.0, 'SB' : 24.0, 'passSvB' : 19.0, 'failSvB' : 14.0, }
        
        
        
        self.counts3_unit = {}
        self.counts3_unit['data_UL18D'] = {'passJetMult' : 1000.0, 'passPreSel' : 975.0, 'passDiJetMass' : 291.0, 'SR' : 113.0, 'SB' : 178.0, 'passSvB' : 1.0, 'failSvB' : 644.0, }
        self.counts3_unit['data_UL18C'] = {'passJetMult' : 1002.0, 'passPreSel' : 965.0, 'passDiJetMass' : 296.0, 'SR' : 116.0, 'SB' : 180.0, 'passSvB' : 0.0, 'failSvB' : 613.0, }
        self.counts3_unit['data_UL18B'] = {'passJetMult' : 1000.0, 'passPreSel' : 953.0, 'passDiJetMass' : 297.0, 'SR' : 128.0, 'SB' : 169.0, 'passSvB' : 0.0, 'failSvB' : 608.0, }
        self.counts3_unit['data_UL18A'] = {'passJetMult' : 1001.0, 'passPreSel' : 956.0, 'passDiJetMass' : 309.0, 'SR' : 131.0, 'SB' : 178.0, 'passSvB' : 0.0, 'failSvB' : 605.0, }
        self.counts3_unit['data_UL17F'] = {'passJetMult' : 1001.0, 'passPreSel' : 961.0, 'passDiJetMass' : 323.0, 'SR' : 127.0, 'SB' : 196.0, 'passSvB' : 1.0, 'failSvB' : 587.0, }
        self.counts3_unit['data_UL17E'] = {'passJetMult' : 1001.0, 'passPreSel' : 957.0, 'passDiJetMass' : 330.0, 'SR' : 109.0, 'SB' : 221.0, 'passSvB' : 1.0, 'failSvB' : 594.0, }
        self.counts3_unit['data_UL17D'] = {'passJetMult' : 1001.0, 'passPreSel' : 945.0, 'passDiJetMass' : 335.0, 'SR' : 138.0, 'SB' : 197.0, 'passSvB' : 0.0, 'failSvB' : 576.0, }
        self.counts3_unit['data_UL17C'] = {'passJetMult' : 1000.0, 'passPreSel' : 945.0, 'passDiJetMass' : 336.0, 'SR' : 118.0, 'SB' : 218.0, 'passSvB' : 1.0, 'failSvB' : 577.0, }
        self.counts3_unit['data_UL16_preVFPE'] = {'passJetMult' : 1000.0, 'passPreSel' : 977.0, 'passDiJetMass' : 382.0, 'SR' : 153.0, 'SB' : 229.0, 'passSvB' : 0.0, 'failSvB' : 622.0, }
        self.counts3_unit['data_UL16_preVFPD'] = {'passJetMult' : 1001.0, 'passPreSel' : 973.0, 'passDiJetMass' : 406.0, 'SR' : 164.0, 'SB' : 242.0, 'passSvB' : 1.0, 'failSvB' : 630.0, }
        self.counts3_unit['data_UL16_preVFPC'] = {'passJetMult' : 1002.0, 'passPreSel' : 974.0, 'passDiJetMass' : 440.0, 'SR' : 161.0, 'SB' : 279.0, 'passSvB' : 1.0, 'failSvB' : 607.0, }
        self.counts3_unit['data_UL16_preVFPB'] = {'passJetMult' : 1000.0, 'passPreSel' : 972.0, 'passDiJetMass' : 398.0, 'SR' : 148.0, 'SB' : 250.0, 'passSvB' : 0.0, 'failSvB' : 624.0, }
        self.counts3_unit['data_UL16_postVFPH'] = {'passJetMult' : 1000.0, 'passPreSel' : 967.0, 'passDiJetMass' : 431.0, 'SR' : 176.0, 'SB' : 255.0, 'passSvB' : 0.0, 'failSvB' : 611.0, }
        self.counts3_unit['data_UL16_postVFPG'] = {'passJetMult' : 1000.0, 'passPreSel' : 970.0, 'passDiJetMass' : 417.0, 'SR' : 151.0, 'SB' : 266.0, 'passSvB' : 0.0, 'failSvB' : 614.0, }
        self.counts3_unit['data_UL16_postVFPF'] = {'passJetMult' : 999.0, 'passPreSel' : 980.0, 'passDiJetMass' : 421.0, 'SR' : 143.0, 'SB' : 278.0, 'passSvB' : 0.0, 'failSvB' : 663.0, }
        self.counts3_unit['ZZ4b_UL18'] = {'passJetMult' : 946.0, 'passPreSel' : 824.0, 'passDiJetMass' : 527.0, 'SR' : 330.0, 'SB' : 197.0, 'passSvB' : 3.0, 'failSvB' : 178.0, }
        self.counts3_unit['ZZ4b_UL17'] = {'passJetMult' : 951.0, 'passPreSel' : 822.0, 'passDiJetMass' : 528.0, 'SR' : 328.0, 'SB' : 200.0, 'passSvB' : 4.0, 'failSvB' : 185.0, }
        self.counts3_unit['ZZ4b_UL16_preVFP'] = {'passJetMult' : 966.0, 'passPreSel' : 856.0, 'passDiJetMass' : 575.0, 'SR' : 338.0, 'SB' : 237.0, 'passSvB' : 6.0, 'failSvB' : 313.0, }
        self.counts3_unit['ZZ4b_UL16_postVFP'] = {'passJetMult' : 972.0, 'passPreSel' : 869.0, 'passDiJetMass' : 570.0, 'SR' : 344.0, 'SB' : 226.0, 'passSvB' : 10.0, 'failSvB' : 270.0, }
        self.counts3_unit['ZH4b_UL18'] = {'passJetMult' : 951.0, 'passPreSel' : 777.0, 'passDiJetMass' : 587.0, 'SR' : 377.0, 'SB' : 210.0, 'passSvB' : 8.0, 'failSvB' : 139.0, }
        self.counts3_unit['ZH4b_UL17'] = {'passJetMult' : 947.0, 'passPreSel' : 760.0, 'passDiJetMass' : 553.0, 'SR' : 362.0, 'SB' : 191.0, 'passSvB' : 17.0, 'failSvB' : 135.0, }
        self.counts3_unit['ZH4b_UL16_preVFP'] = {'passJetMult' : 955.0, 'passPreSel' : 807.0, 'passDiJetMass' : 616.0, 'SR' : 403.0, 'SB' : 213.0, 'passSvB' : 22.0, 'failSvB' : 205.0, }
        self.counts3_unit['ZH4b_UL16_postVFP'] = {'passJetMult' : 956.0, 'passPreSel' : 785.0, 'passDiJetMass' : 605.0, 'SR' : 435.0, 'SB' : 170.0, 'passSvB' : 30.0, 'failSvB' : 175.0, }
        self.counts3_unit['TTToSemiLeptonic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 988.0, 'passDiJetMass' : 648.0, 'SR' : 332.0, 'SB' : 316.0, 'passSvB' : 3.0, 'failSvB' : 358.0, }
        self.counts3_unit['TTToSemiLeptonic_UL17'] = {'passJetMult' : 1000.0, 'passPreSel' : 991.0, 'passDiJetMass' : 621.0, 'SR' : 323.0, 'SB' : 298.0, 'passSvB' : 2.0, 'failSvB' : 381.0, }
        self.counts3_unit['TTToSemiLeptonic_UL16_preVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 989.0, 'passDiJetMass' : 616.0, 'SR' : 321.0, 'SB' : 295.0, 'passSvB' : 1.0, 'failSvB' : 440.0, }
        self.counts3_unit['TTToSemiLeptonic_UL16_postVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 987.0, 'passDiJetMass' : 645.0, 'SR' : 332.0, 'SB' : 313.0, 'passSvB' : 5.0, 'failSvB' : 452.0, }
        self.counts3_unit['TTToHadronic_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 991.0, 'passDiJetMass' : 738.0, 'SR' : 419.0, 'SB' : 319.0, 'passSvB' : 2.0, 'failSvB' : 268.0, }
        self.counts3_unit['TTToHadronic_UL17'] = {'passJetMult' : 1000.0, 'passPreSel' : 993.0, 'passDiJetMass' : 737.0, 'SR' : 409.0, 'SB' : 328.0, 'passSvB' : 1.0, 'failSvB' : 250.0, }
        self.counts3_unit['TTToHadronic_UL16_preVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 993.0, 'passDiJetMass' : 734.0, 'SR' : 397.0, 'SB' : 337.0, 'passSvB' : 0.0, 'failSvB' : 300.0, }
        self.counts3_unit['TTToHadronic_UL16_postVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 992.0, 'passDiJetMass' : 728.0, 'SR' : 395.0, 'SB' : 333.0, 'passSvB' : 1.0, 'failSvB' : 302.0, }
        self.counts3_unit['TTTo2L2Nu_UL18'] = {'passJetMult' : 1001.0, 'passPreSel' : 977.0, 'passDiJetMass' : 504.0, 'SR' : 240.0, 'SB' : 264.0, 'passSvB' : 4.0, 'failSvB' : 496.0, }
        self.counts3_unit['TTTo2L2Nu_UL17'] = {'passJetMult' : 1000.0, 'passPreSel' : 972.0, 'passDiJetMass' : 495.0, 'SR' : 231.0, 'SB' : 264.0, 'passSvB' : 3.0, 'failSvB' : 489.0, }
        self.counts3_unit['TTTo2L2Nu_UL16_preVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 985.0, 'passDiJetMass' : 467.0, 'SR' : 220.0, 'SB' : 247.0, 'passSvB' : 0.0, 'failSvB' : 564.0, }
        self.counts3_unit['TTTo2L2Nu_UL16_postVFP'] = {'passJetMult' : 1000.0, 'passPreSel' : 975.0, 'passDiJetMass' : 469.0, 'SR' : 210.0, 'SB' : 259.0, 'passSvB' : 1.0, 'failSvB' : 574.0, }
        self.counts3_unit['HH4b_UL18'] = {'passJetMult' : 978.0, 'passPreSel' : 760.0, 'passDiJetMass' : 432.0, 'SR' : 307.0, 'SB' : 125.0, 'passSvB' : 26.0, 'failSvB' : 222.0, }
        self.counts3_unit['HH4b_UL17'] = {'passJetMult' : 974.0, 'passPreSel' : 763.0, 'passDiJetMass' : 441.0, 'SR' : 306.0, 'SB' : 135.0, 'passSvB' : 15.0, 'failSvB' : 205.0, }
        self.counts3_unit['HH4b_UL16_preVFP'] = {'passJetMult' : 982.0, 'passPreSel' : 799.0, 'passDiJetMass' : 433.0, 'SR' : 292.0, 'SB' : 141.0, 'passSvB' : 16.0, 'failSvB' : 226.0, }
        


        

        
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
