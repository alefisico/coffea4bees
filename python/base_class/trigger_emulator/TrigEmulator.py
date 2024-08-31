class TrigEmulator:
    def __init__(self, ht_thresholds, jet_thresholds, jet_multiplicities, btag_op_points, btag_multiplicities):
        self.m_htThresholds = ht_thresholds
        self.m_jetThresholds = jet_thresholds
        self.m_jetMultiplicities = jet_multiplicities
        self.m_bTagOpPoints = btag_op_points
        self.m_bTagMultiplicities = btag_multiplicities

    def passTrig(self, offline_jet_pts, offline_btagged_jet_pts, ht, seedOffset):
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


    def passTrigCorrelated(self, offline_jet_pts, offline_btagged_jet_pts, ht, btag_rand, ht_rand, seedOffset):
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


    def calcWeight(self, offline_jet_pts, offline_btagged_jet_pts, ht):
        nPass = 0

        for iToy in range(self.m_nToys):
            # Count all events
            if self.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, iToy):
                nPass += 1

        weight = float(nPass) / self.m_nToys
        # print(f"TrigEmulator::calcWeight is {weight}")
        return weight


    def Fill(self, offline_jet_pts, offline_btagged_jet_pts, ht):
        for iToy in range(self.m_nToys):
            # Count all events
            self.m_nTotal += 1
            if self.passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, iToy):
                self.m_nPass += 1

        return


#
# # Example instantiation
# emulator = TrigEmulator(nToys=1000)
#
# # Example data
# offline_jet_pts = [100, 150, 200]
# offline_btagged_jet_pts = [120, 180]
# ht = 500
#
# # Calculate the weight
# weight = emulator.calcWeight(offline_jet_pts, offline_btagged_jet_pts, ht)
# print(weight)  # Output: The calculated weight as a float
