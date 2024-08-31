import numpy as np

class HLTJetEmulator:
    def __init__(self, high_bin_edge, eff, eff_err):
        self.m_highBinEdge = high_bin_edge
        self.m_eff = eff
        self.m_effErr = eff_err
        self.m_rand = np.random.default_rng()  # Initialize a random number generator

    def passJet(self, pt, seedOffset=1.0, smearFactor=0.0):
        eff = -99
        effErr = -99

        for iBin in range(len(self.m_highBinEdge)):
            if pt < self.m_highBinEdge[iBin]:
                eff = self.m_eff[iBin]
                effErr = self.m_effErr[iBin]
                break

        if eff < -90:
            eff = self.m_eff[-1]
            effErr = self.m_effErr[-1]

        if eff < 0:
            eff = 0

        thisTagEff = eff + effErr * smearFactor
        seed = int(pt * seedOffset + pt)
        # np.random.seed(seed)  # Uncomment if you want to set a seed based on pt and seedOffset
        # self.m_rand = np.random.default_rng(seed)  # Alternative way to set the seed

        if thisTagEff > self.m_rand.random():  # Generate a random number in [0, 1)
            return True

        return False
