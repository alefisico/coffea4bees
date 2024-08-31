import random
import yaml

from ..trigger_emulator.HLTBTagEmulator    import HLTBTagEmulator
from ..trigger_emulator.HLTHtEmulator      import HLTHtEmulator
from ..trigger_emulator.HLTJetEmulator     import HLTJetEmulator


class TrigEmulatorTool:
    def __init__(self, name, nToys=1000, year="2018", debug=False, useMCTurnOns=False):
        self.m_name = name
        self.m_nToys = nToys
        self.m_debug = debug
        self.m_useMCTurnOns = useMCTurnOns
        self.m_rand = random.Random()

        self.m_Jet = {}
        self.m_Ht = {}
        self.m_BTag = {}

        if year == "2018":
            self.config2018()
        elif year == "2017":
            self.config2017()
        elif year == "2016":
            self.config2016()
        else:
            print(f"TrigEmulatorTool::ERROR year has to be 2018, 2017 or 2016. Not {year}")

        if self.m_debug:
            print("TrigEmulatorTool::Making Jet Thresholds")
        for trig_name, trig_data in self.JetConfig.items():
            print(trig_name, trig_data.keys())
            self.m_Jet[trig_name] = HLTJetEmulator(high_bin_edge=trig_data["high_bin_edge"] , eff=trig_data["eff"], eff_err=trig_data["eff_err"])

        if self.m_debug:
            print("TrigEmulatorTool::Making Ht Thresholds")
        for hc in self.HTConfig:
            self.m_Ht[hc[0]] = HLTHtEmulator(hc[1][0], hc[1][1], debug)

        if self.m_debug:
            print("TrigEmulatorTool::Making BTag Thresholds")
        for bc in self.BTagConfig:
            self.m_BTag[bc[0]] = HLTBTagEmulator(bc[1][0], bc[1][1], debug)

        if self.m_debug:
            print("TrigEmulatorTool::Leaving")

    def AddTrig(self, trigName, HTNames=None, JetNames=None, JetMults=None, TagNames=None, TagMults=None):
        if self.m_debug:
            print(f"TrigEmulatorTool::AddTrig Enter {trigName}")

        if trigName in self.m_emulatedTrigMenu:
            print(f"TrigEmulatorTool::AddTrig ERROR {trigName} already included")
            return

        if HTNames is None:
            HTNames = []
        if JetNames is None:
            JetNames = []
        if JetMults is None:
            JetMults = []
        if TagNames is None:
            TagNames = []
        if TagMults is None:
            TagMults = []

        assert len(JetNames) == len(JetMults)
        assert len(TagNames) == len(TagMults)
        assert len(TagNames) < 3

        HTCuts = [self.m_Ht[ht] for ht in HTNames if ht in self.m_Ht]
        JetPtCuts = [self.m_Jet[jt] for jt in JetNames if jt in self.m_Jet]
        BTagPtCuts = [self.m_BTag[bt] for bt in TagNames if bt in self.m_BTag]

        if self.m_debug:
            print(f"TrigEmulatorTool::AddTrig inserting {trigName}")

        self.m_emulatedTrigMenu[trigName] = TrigEmulator(trigName, HTCuts, JetPtCuts, JetMults, BTagPtCuts, TagMults, self.m_nToys)
        self.m_emulatedDecisions[trigName] = False
        self.m_emulatedWeights[trigName] = 0.0

    #   Sets decisions for all configured triggers (involves random numbers)
    def SetDecisions(self, offline_jet_pts, offline_btagged_jet_pts, ht=-1):
        for trigName, trigEmulator in self.m_emulatedTrigMenu.items():
            self.m_emulatedDecisions[trigName] = trigEmulator.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht)

    # Return the value set in SetDecisions.  (So must call SetDecisions before GetDecision/Passed)
    def GetDecision(self, trigName):
        if trigName not in self.m_emulatedDecisions:
            print(f"TrigEmulatorTool::GetDecisions ERROR {trigName} not defined. Returning false")
            return False
        return self.m_emulatedDecisions[trigName]

    #  Sets weights for all configured triggers, which is the average nPass over nToys (involves random numbers)
    def SetWeights(self, offline_jet_pts, offline_btagged_jet_pts, ht=-1):
        for trigName, trigEmulator in self.m_emulatedTrigMenu.items():
            self.m_emulatedWeights[trigName] = trigEmulator.calcWeight(offline_jet_pts, offline_btagged_jet_pts, ht)

    #    Return the value set in SetWeights.  (So must call SetWeights before GetWeight)
    def GetWeight(self, trigName):
        if trigName not in self.m_emulatedWeights:
            print(f"TrigEmulatorTool::GetWeight ERROR {trigName} not defined. Returning 0.0")
            return 0.0
        return self.m_emulatedWeights[trigName]

    #  Calculate the weight of the OR of the menu defined
    def GetWeightOR(self, offline_jet_pts, offline_btagged_jet_pts, ht=-1, setSeed=False):
        nPass = 0

        for iToy in range(self.m_nToys):
            passAny = False
            btag_weights = self.getRandWeights(offline_btagged_jet_pts, setSeed, 2 * iToy)

            if setSeed:
                seed = int(ht * (3 * iToy) + ht)
                self.m_rand.seed(seed)

            ht_weights = [self.m_rand.random() for _ in range(3)]

            for trigName, trigEmulator in self.m_emulatedTrigMenu.items():
                if trigEmulator.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, btag_weights, ht_weights, iToy):
                    passAny = True

            if passAny:
                nPass += 1

        return float(nPass) / self.m_nToys

    def Fill(self, offline_jet_pts, offline_btagged_jet_pts, ht):
        for trigEmulator in self.m_emulatedTrigMenu.values():
            trigEmulator.Fill(offline_jet_pts, offline_btagged_jet_pts, ht)

    def dumpResults(self):
        for trigName, trigEmulator in self.m_emulatedTrigMenu.items():
            print(trigName, end=" ")
            trigEmulator.dumpResults()


    def config2018(self):
        print("Config 2018")

        self.JetConfig  = {}
        self.HTConfig   = {}
        self.BTagConfig = {}

        # Configuration for 2018
        input_file_name = "base_class/trigger_emulator/data/haddOutput_All_MC2018_11Nov_fittedTurnOns.yaml"
        with open(input_file_name, 'r') as infile:
            data = yaml.safe_load(infile)

        self.JetConfig["jetTurnOn::PF30BTag"] = data["pt_s_PF30inMJMatchBtagTandP_jetID_TurnOn"]
# jetTurnOn::PF30BTag,                   {{"pt_s_PF30inMJMatchBtagTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF40BTag,                   {{"pt_s_PF40inMJMatchBtagTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF45BTag,                   {{"pt_s_PF45inMJMatchBtagTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF60BTag,                   {{"pt_PF60inMJMatchBtagTandP_jetID_TurnOn",                       fileName2018}} },
# jetTurnOn::PF75BTag,                   {{"pt_PF75inMJMatchBtagTandP_jetID_TurnOn",                       fileName2018}} },
# jetTurnOn::PF116BTag,                  {{"pt_PF116inMJMatchBtagTandP_jetID_TurnOn",                      fileName2018}} },
# jetTurnOn::PF116DrBTag,                {{"pt_PF116DrfilterMatchBtagTandP_jetID_TurnOn",                    fileName2018}} },
# jetTurnOn::L1112BTag,   {{"pt_L12b112inMJMatchBtagTandP_jetID_TurnOn", fileName2018}} },
# jetTurnOn::Calo100BTag, {{"pt_Calo100inMJMatchBtagTandP_jetID_TurnOn", fileName2018}} },
# jetTurnOn::PF30,                   {{"pt_s_PF30inMJTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF40,                   {{"pt_s_PF40inMJTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF45,                   {{"pt_s_PF45inMJTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF60,                   {{"pt_PF60inMJTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF75,                   {{"pt_PF75inMJTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::PF116,                  {{"pt_PF116inMJTandP_jetID_TurnOn",                     fileName2018}} },
# jetTurnOn::L1112,   {{"pt_L12b112inMJTandP_jetID_TurnOn", fileName2018}} },
# jetTurnOn::Calo100, {{"pt_Calo100inMJTandP_jetID_TurnOn", fileName2018}} },
        #print(data.keys())


    def config2017(self):
        # Configuration for 2017
        pass

    def config2016(self):
        # Configuration for 2016
        pass

    # getRandWeights method needs to be implemented
    def getRandWeights(self, offline_btagged_jet_pts, setSeed, seed):
        # Implement the method to generate random weights
        pass
