import numpy as np


class HLTBTagEmulator:
    def __init__(self, high_bin_edge, eff, eff_err):
        self.m_highBinEdge = high_bin_edge
        self.m_eff = eff
        self.m_effErr = eff_err
        self.m_rand = np.random.default_rng()  # Initialize a random number generator

        #
        #  Yaml file loading...
        #



    def passJetThreshold(self, pt, bTagRand, smearFactor=0.0):
        eff = -99
        effErr = -99

        for iBin in range(len(self.m_highBinEdge)):
            if pt < self.m_highBinEdge[iBin]:
                eff = self.m_eff[iBin]
                effErr = self.m_effErr[iBin]
                break

        if eff < 0:
            eff = self.m_eff[-1]
            effErr = self.m_effErr[-1]
        if eff < 0:
            eff = 0

        thisTagEff = eff + effErr * smearFactor
        if thisTagEff > bTagRand:
            return True

        return False

    def passJet(self, pt, seedOffset=1.0, smearFactor=0.0):
        # Optionally set the seed, similar to the C++ code (commented out here)
        # seed = int(pt * seedOffset + pt)
        # np.random.seed(seed)  # Set seed for reproducibility, if needed

        bTagRand = self.m_rand.random()  # Generate a random number in [0, 1)
        return self.passJetThreshold(pt, bTagRand, smearFactor)
