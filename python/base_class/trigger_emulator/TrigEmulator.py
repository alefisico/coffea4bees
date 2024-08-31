class TrigEmulator:
    def __init__(self, ht_thresholds, jet_thresholds, jet_multiplicities, btag_op_points, btag_multiplicities, nToys=100):
        self.m_htThresholds = ht_thresholds
        self.m_jetThresholds = jet_thresholds
        self.m_jetMultiplicities = jet_multiplicities
        self.m_bTagOpPoints = btag_op_points
        self.m_bTagMultiplicities = btag_multiplicities
        self.m_nToys = nToys

    def passTrig(self, offline_jet_pts, offline_btagged_jet_pts, ht=-1, seedOffset=1.0):
        # Ht Cut
        for iThres in range(len(self.m_htThresholds)):
            HLTHtCut = self.m_htThresholds[iThres]

            if ht > 0 and HLTHtCut:
                if not HLTHtCut.passHt(ht, seedOffset):
                    return False

        # Loop on all thresholds
        for iThres in range(len(self.m_jetThresholds)):
            HLTJet = self.m_jetThresholds[iThres]
            nJetsPassed = 0

            # Count passing jets
            for jet_pt in offline_jet_pts:
                if HLTJet.passJet(jet_pt, seedOffset):
                    nJetsPassed += 1

            # Impose trigger cut
            if nJetsPassed < self.m_jetMultiplicities[iThres]:
                return False

        # Apply BTag Operating Points
        for iThres in range(len(self.m_bTagOpPoints)):
            HLTBTag = self.m_bTagOpPoints[iThres]
            nJetsPassBTag = 0

            # Count passing jets
            for bjet_pt in offline_btagged_jet_pts:
                if HLTBTag.passJet(bjet_pt, seedOffset):
                    nJetsPassBTag += 1

            # Impose trigger cut
            if nJetsPassBTag < self.m_bTagMultiplicities[iThres]:
                return False

        return True

    # Used for calculating correlated decisions with input (ht and btagging) weights
    def passTrigCorrelated(self, offline_jet_pts, offline_btagged_jet_pts, ht, btag_rand, ht_rand, seedOffset=1.0):
        # Ht Cut
        for iThres in range(len(self.m_htThresholds)):
            HLTHtCut = self.m_htThresholds[iThres]

            if ht > 0 and HLTHtCut:
                if not HLTHtCut.passHtThreshold(ht, ht_rand[iThres]):
                    return False

        # Loop on all thresholds
        for iThres in range(len(self.m_jetThresholds)):
            HLTJet = self.m_jetThresholds[iThres]
            nJetsPassed = 0

            # Count passing jets
            for jet_pt in offline_jet_pts:
                if HLTJet.passJet(jet_pt, seedOffset):
                    nJetsPassed += 1

            # Impose trigger cut
            if nJetsPassed < self.m_jetMultiplicities[iThres]:
                return False

        # Apply BTag Operating Points
        for iThres in range(len(self.m_bTagOpPoints)):
            HLTBTag = self.m_bTagOpPoints[iThres]
            nJetsPassBTag = 0

            # Count passing jets
            for iBJet in range(len(offline_btagged_jet_pts)):
                bjet_pt = offline_btagged_jet_pts[iBJet]
                if HLTBTag.passJetThreshold(bjet_pt, btag_rand[iBJet][iThres]):
                    nJetsPassBTag += 1

            # Impose trigger cut
            if nJetsPassBTag < self.m_bTagMultiplicities[iThres]:
                return False

        return True

    #  Calculate weight for trigger, average nPass over nToys
    def calcWeight(self, offline_jet_pts, offline_btagged_jet_pts, ht=-1):
        nPass = 0

        for iToy in range(self.m_nToys):
            # Count all events
            if self.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, iToy):
                nPass += 1

        weight = float(nPass) / self.m_nToys
        # print(f"TrigEmulator::calcWeight is {weight}")
        return weight

    # #  For doing global run counting (Eg: in rate prediction)
    # def Fill(self, offline_jet_pts, offline_btagged_jet_pts, ht):
    #     for iToy in range(self.m_nToys):
    #         # Count all events
    #         self.m_nTotal += 1
    #         if self.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, iToy):
    #             self.m_nPass += 1
    #
    #     return
