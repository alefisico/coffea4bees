import awkward as ak

from numba import jit



#@jit(nopython=True)
def buildTops(input_jets):
    #
    #  Sort Jets
    #
    #print(f"{input_jets.pt}")
    #print(f"{input_jets.btagDeepFlavB}")

    #print(f"{input_jets.pt}")
    #print(f"{input_jets.btagDeepFlavB}")

    #
    #  Jet collections
    #
    topQuarkBJets = input_jets[0:3]
    topQuarkWJets = input_jets[2:]
    #print(f"{topQuarkBJets}")
    #print(f"{topQuarkWJets}")

    out_xW = []
    out_xbW = []
    
    # Based on  https://github.com/patrickbryant/ZZ4b/blob/master/nTupleAnalysis/src/eventData.cc#L1333
    for b in topQuarkBJets:
        for j in topQuarkWJets:
            if bool(b.mass == j.mass) and bool(b.pt == j.pt): 
                continue #require they are different jets
            if b.btagDeepFlavB < j.btagDeepFlavB:
                continue  #don't consider W pairs where j is more b-like than b.

            for l in topQuarkWJets:
                if bool(b.mass == l.mass) and bool(b.pt == l.pt): 
                    continue
                if bool(j.mass == l.mass) and bool(j.pt == l.pt): 
                    continue
                if j.btagDeepFlavB < l.btagDeepFlavB:
                    continue  #don't consider W pairs where l is more b-like than j.

                #print(f"Trying tri-jet {b} and {j} and {l}")

                #            trijet* thisTop = new trijet(b,j,l);
                # Based on https://github.com/patrickbryant/nTupleAnalysis/blob/master/baseClasses/src/trijet.cc
                #
                bReg = b * b.bRegCorr
                jReg = j #* j.bRegCorr
                lReg = l # l.bRegCorr

                # https://github.com/patrickbryant/nTupleAnalysis/blob/master/baseClasses/src/dijet.cc
                dijet_jl = jReg + lReg
                
                mW =  80.4;
                xW = (dijet_jl.mass-mW)/(0.10*dijet_jl.mass) 

                dijet_jl_wCor  = dijet_jl*(mW/dijet_jl.mass);
                mbW  = (bReg + dijet_jl_wCor).mass

                mt = 173.0;
                xbW  = (mbW-mt)/(0.05*mbW) #smaller resolution term because there are fewer degrees of freedom. FWHM=25GeV, about the same as mW 
                #xbW = thisTop->xbW;
                out_xbW.append(xbW)
                out_xW.append(xW)

    return out_xW, out_xbW
