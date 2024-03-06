from scipy.special import comb
import numpy as np

def getCombinatoricWeightOld(nj, f, e=0.0, d=1.0, norm=1.0):
    w = 0
    nbt = 3 #number of required bTags
    nlt = nj-nbt #number of selected untagged jets ("light" jets)
    nPseudoTagProb = np.zeros(nlt+1)
    for npt in range(0,nlt + 1):#npt is the number of pseudoTags in this combination
        nt = nbt + npt
        nnt = nlt-npt # number of not tagged
        # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
        w_npt = norm * comb(nlt,npt, exact=True) * f**npt * (1-f)**nnt
        if (nt%2) == 0: w_npt *= 1 + e/nlt**d

        nPseudoTagProb[npt] += w_npt
    w = np.sum(nPseudoTagProb[1:])
    return w, nPseudoTagProb




def getPseudoTagProbs(nj, f, e=0.0, d=1.0, norm=1.0):
    nbt = 3 #number of required bTags
    nlt = nj-nbt #number of selected untagged jets ("light" jets)
    nPseudoTagProb = np.zeros(nlt+1)
    for npt in range(0,nlt + 1):#npt is the number of pseudoTags in this combination
        nt = nbt + npt
        nnt = nlt-npt # number of not tagged
        # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
        w_npt = norm * comb(nlt,npt, exact=True) * f**npt * (1-f)**nnt
        if (nt%2) == 0: w_npt *= 1 + e/nlt**d

        nPseudoTagProb[npt] += w_npt
    return nPseudoTagProb



def getCombinatoricWeight(nj, f, e=0.0, d=1.0, norm=1.0):
    nPseudoTagProb = getPseudoTagProbs(nj, f, e, d, norm)
    return np.sum(nPseudoTagProb[1:])
#    nbt = 3 #number of required bTags
#    nlt = nj-nbt #number of selected untagged jets ("light" jets)
#    nPseudoTagProb = np.zeros(nlt+1)
#    for npt in range(0,nlt + 1):#npt is the number of pseudoTags in this combination
#        nt = nbt + npt
#        nnt = nlt-npt # number of not tagged
#        # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
#        w_npt = norm * comb(nlt,npt, exact=True) * f**npt * (1-f)**nnt
#        if (nt%2) == 0: w_npt *= 1 + e/nlt**d
#
#        nPseudoTagProb[npt] += w_npt
#    w = np.sum(nPseudoTagProb[1:])
#    return w, nPseudoTagProb
