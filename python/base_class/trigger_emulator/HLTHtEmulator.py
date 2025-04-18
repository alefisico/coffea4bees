import numpy as np
import logging

class HLTHtEmulator:
    def __init__(self, high_bin_edge, eff, eff_err):
        self.m_highBinEdge = high_bin_edge
        self.m_eff = eff
        self.m_effErr = eff_err
        self.m_rand = np.random.default_rng()  # Initialize a random number generator
        self.m_nBins = len(self.m_highBinEdge)

    def passHt(self, ht, seedOffset=1.0, smearFactor=0.0):
        # Optionally set the seed, similar to the C++ code (commented out here)
        # seed = int(ht * seedOffset + ht)
        # np.random.seed(seed)  # Set seed for reproducibility, if needed

        htRand = self.m_rand.random()  # Generate a random number in [0, 1)
        return self.passHtThreshold(ht, htRand, smearFactor)

    def passHtThreshold(self, ht, htRand, smearFactor=0.0, debug=False):

        eff = -99
        effErr = -99
        if debug: print(f" HLTHtEmulator.passHtThreshold m_nBins = {self.m_nBins}")
        for iBin in range(self.m_nBins):
            if debug:
                print(f"{iBin} comparing {ht} to {self.m_highBinEdge[iBin]} ")

            if ht < self.m_highBinEdge[iBin]:
                eff = self.m_eff[iBin]
                effErr = self.m_effErr[iBin]
                if debug:
                    logging.debug(f"eff is {eff}")
                break

        if eff < 0:
            eff = self.m_eff[-1]
            effErr = self.m_effErr[-1]

        assert eff >= 0, "ERROR: eff < 0"

        thisTagEff = eff + effErr * smearFactor
        if debug:
            logging.debug(f"thisTagEff {thisTagEff} for ht = {ht}")
        return thisTagEff > htRand
