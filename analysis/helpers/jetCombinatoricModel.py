import awkward as ak
import numpy as np

class jetCombinatoricModel:
    def __init__(self, filename, cut='passPreSel'):
        self.filename = filename
        self.cut = cut
        self.read_parameter_file()
        # print(self.data)

    def read_parameter_file(self):
        self.data = {}
        with open(self.filename, 'r') as lines:
            for line in lines:
                words = line.split()
                if not len(words): continue
                if len(words) == 2:
                    self.data[words[0]] = float(words[1])
                else:
                    self.data[words[0]] = ' '.join(words[1:])

        self.p = self.data[f'pseudoTagProb_{self.cut}']
        self.e = self.data[f'pairEnhancement_{self.cut}']
        self.d = self.data[f'pairEnhancementDecay_{self.cut}']
        self.t = self.data[f'threeTightTagFraction_{self.cut}']

    def __call__(self, untagged_jets):
        nEvent = len(untagged_jets)
        maxPseudoTags = 12
        nbt = 3 # number of required b-tags
        nlt = ak.to_numpy( ak.num(untagged_jets, axis=1) ) # number of light jets
        nPseudoTagProb = np.zeros((maxPseudoTags+1, nEvent))
        nPseudoTagProb[0] = self.t * (1-self.p)**nlt
        for npt in range(1,maxPseudoTags+1): # iterate over all possible number of pseudo-tags
            nt  = nbt + npt # number of tagged jets (b-tagged or pseudo-tagged)
            nnt = nlt - npt # number of not tagged jets (b-tagged or pseudo-tagged)
            nnt[nnt<0] = 0 # in cases where npt>nlt, set nnt to zero
            ncr = ak.to_numpy( ak.num(ak.combinations(untagged_jets, npt)) ) # number of ways to get npt pseudo-tags
            w_npt = self.t * ncr * self.p**npt * (1-self.p)**nnt
            if (nt%2)==0: # event number of tags boost from pair production enhancement term
                w_npt *= 1 + self.e/nlt**self.d
            nPseudoTagProb[npt] = w_npt
        w = np.sum(nPseudoTagProb[1:], axis=0)
        r = np.random.uniform(0,1, size=nEvent)*w + nPseudoTagProb[0] # random number between nPseudoTagProb[0] and nPseudoTagProb.sum(axis=0)
        r = r.reshape(1,nEvent).repeat(maxPseudoTags+1,0)
        c = np.array([nPseudoTagProb[:npt+1].sum(axis=0) for npt in range(maxPseudoTags+1)]) # cumulative prob
        npt = (r>c).sum(axis=0)
        return w, npt

