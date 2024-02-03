import awkward as ak


import numba
from math import sqrt


#@jit(nopython=True)
def buildTops_single(input_jets):
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
    out_xWbW = []
    
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

                xWbW = sqrt(pow(xW, 2) + pow(xbW,2));
                #xbW = thisTop->xbW;
                out_xbW.append(xbW)
                out_xW.append(xW)
                out_xWbW.append(xWbW)

    return out_xW, out_xbW, out_xWbW




@numba.njit
def find_tops_kernel(events_jets, builder):
    """Search for valid 4-lepton combinations from an array of events * leptons {charge, ...}

    A valid candidate has two pairs of leptons that each have balanced charge
    Outputs an array of events * candidates {indices 0..3} corresponding to all valid
    permutations of all valid combinations of unique leptons in each event
    (omitting permutations of the pairs)
    """
    for jets in events_jets:
        #print(f"jets.pt are {jets.pt}\n")
        #print(f"jets.btagDeepFlavB are {jets.btagDeepFlavB}\n")
        builder.begin_list()
        nJets = len(jets)
        for ib in range(0, 3):
            for ij in range(2, 4):
                if len({ib, ij}) < 2:
                    continue

                if jets[ib].btagDeepFlavB < jets[ij].btagDeepFlavB:
                    continue  #don't consider W pairs where j is more b-like than b.

                for il in range(2, 4):
                        if len({ib, ij, il}) < 3:
                            continue
                        if jets[ij].btagDeepFlavB < jets[il].btagDeepFlavB:
                            continue  #don't consider W pairs where l is more b-like than j.

                        builder.begin_tuple(3)
                        builder.index(0).integer(ib)
                        builder.index(1).integer(ij)
                        builder.index(2).integer(il)
                        builder.end_tuple()

        builder.end_list()

    return builder


def find_tops_kernel_slow(events_jets, builder):
    """Search for valid 4-lepton combinations from an array of events * leptons {charge, ...}

    A valid candidate has two pairs of leptons that each have balanced charge
    Outputs an array of events * candidates {indices 0..3} corresponding to all valid
    permutations of all valid combinations of unique leptons in each event
    (omitting permutations of the pairs)
    """
    for jets in events_jets:
        builder.begin_list()
        nJets = len(jets)
        for ib in range(0, 3):
            for ij in range(2, 4):
                if len({ib, ij}) < 2:
                    continue

                if jets[ib].btagDeepFlavB < jets[ij].btagDeepFlavB:
                    continue  #don't consider W pairs where j is more b-like than b.

                for il in range(2, 4):
                        if len({ib, ij, il}) < 3:
                            continue
                        if jets[ij].btagDeepFlavB < jets[il].btagDeepFlavB:
                            continue  #don't consider W pairs where l is more b-like than j.

                        builder.begin_tuple(3)
                        builder.index(0).integer(ib)
                        builder.index(1).integer(ij)
                        builder.index(2).integer(il)
                        builder.end_tuple()
        builder.end_list()

    return builder



def find_tops(events_jets):

    #if ak.backend(events_jets) == "typetracer":
    #    raise Exception("typetracer")
    #    # here we fake the output of find_4lep_kernel since
    #    # operating on length-zero data returns the wrong layout!
    #    ak.typetracer.length_zero_if_typetracer(events_jets.btagDeepFlavB) # force touching of the necessary data
    #    return ak.Array(ak.Array([[(0,0,0)]]).layout.to_typetracer(forget_length=True))
    return find_tops_kernel(events_jets, ak.ArrayBuilder()).snapshot()


def find_tops_slow(events_jets):
    #if ak.backend(events_leptons) == "typetracer":
    #    # here we fake the output of find_4lep_kernel since
    #    # operating on length-zero data returns the wrong layout!
    #    ak.typetracer.length_zero_if_typetracer(events_leptons.charge) # force touching of the necessary data
    #    return ak.Array(ak.Array([[(0,0,0,0)]]).layout.to_typetracer(forget_length=True))
    return find_tops_kernel_slow(events_jets, ak.ArrayBuilder()).snapshot()


def dumpTopCandidateTestVectors(event, logging, chunk, nEvent):

    for iEvent in range(nEvent):
        logging.info(f'{chunk} event idx ={iEvent} selectedJets pt {event[iEvent].Jet[event[iEvent].Jet.selected].pt}\n')
        logging.info(f'{chunk} event idx ={iEvent} selectedJets eta {event[iEvent].Jet[event[iEvent].Jet.selected].eta}\n')
        logging.info(f'{chunk} event idx ={iEvent} selectedJets phi {event[iEvent].Jet[event[iEvent].Jet.selected].phi}\n')
        logging.info(f'{chunk} event idx ={iEvent} selectedJets mass {event[iEvent].Jet[event[iEvent].Jet.selected].mass}\n')
        logging.info(f'{chunk} event idx ={iEvent} selectedJets btagDeepFlavB {event[iEvent].Jet[event[iEvent].Jet.selected].btagDeepFlavB}\n')
        logging.info(f'{chunk} event idx ={iEvent} selectedJets bRegCorr {event[iEvent].Jet[event[iEvent].Jet.selected].bRegCorr}\n')
        logging.info(f'{chunk} event idx ={iEvent} xbW {event[iEvent].xbW}\n')
        logging.info(f'{chunk} event idx ={iEvent} xW {event[iEvent].xW}\n')


