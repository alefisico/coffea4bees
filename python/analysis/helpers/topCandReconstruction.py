import awkward as ak
import numpy as np
import numba
from math import sqrt


@numba.njit
def find_tops0_kernel(events_jets, builder):
    """Search for top quarks

       All quadjet events will have well defined xWt0, a top candidate where all three jets are allowed to be candidate jets.
    """

    for jets in events_jets:
        #print(f"jets.pt are {jets.pt}\n")
        #print(f"jets.btagDeepFlavB are {jets.btagDeepFlavB}\n")
        builder.begin_list()
        nJets = len(jets)
        for ib in range(0, 3):
            for ij in range(2, nJets):
                if len({ib, ij}) < 2:
                    continue

                if jets[ib].btagDeepFlavB < jets[ij].btagDeepFlavB:
                    continue  #don't consider W pairs where j is more b-like than b.

                for il in range(2, nJets):
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


@numba.njit
def find_tops1_kernel(events_jets, builder):
    """Search for top quarks

    for events with additional jets passing preselection criteria, make top candidates requiring at least one of the jets to be not a candidate jet. 
    This is a way to use b-tagging information without creating a bias in performance between the three and four tag data.
    This should be a higher quality top candidate because W bosons decays cannot produce b-quarks. 
    """

    for jets in events_jets:

        builder.begin_list()
        nJets = len(jets)
        for ib in range(0, 3):  # topQuark BJets
            for ij in range(2, nJets):  # topQuark WJets
                if len({ib, ij}) < 2:
                    continue

                if jets[ib].btagDeepFlavB < jets[ij].btagDeepFlavB:
                    continue  #don't consider W pairs where j is more b-like than b.

                for il in range(4, nJets):
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



def find_tops0(events_jets):

    #if ak.backend(events_jets) == "typetracer":
    #    raise Exception("typetracer")
    #    # here we fake the output of find_4lep_kernel since
    #    # operating on length-zero data returns the wrong layout!
    #    ak.typetracer.length_zero_if_typetracer(events_jets.btagDeepFlavB) # force touching of the necessary data
    #    return ak.Array(ak.Array([[(0,0,0)]]).layout.to_typetracer(forget_length=True))
    return find_tops0_kernel(events_jets, ak.ArrayBuilder()).snapshot()

def find_tops1(events_jets):

    #if ak.backend(events_jets) == "typetracer":
    #    raise Exception("typetracer")
    #    # here we fake the output of find_4lep_kernel since
    #    # operating on length-zero data returns the wrong layout!
    #    ak.typetracer.length_zero_if_typetracer(events_jets.btagDeepFlavB) # force touching of the necessary data
    #    return ak.Array(ak.Array([[(0,0,0)]]).layout.to_typetracer(forget_length=True))
    return find_tops1_kernel(events_jets, ak.ArrayBuilder()).snapshot()



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

def buildTop(input_jets, top_cand_idx):
    """ Takes indices of jets and returns reconstructed top candidate

    """
    top_cands0 = [input_jets[top_cand_idx[idx]] for idx in "012"]
    rec_top_cands = ak.zip({
        "w": ak.zip({
            "jl": top_cands0[1] + top_cands0[2],
        }),
            "bReg": top_cands0[0]* top_cands0[0].bRegCorr
    })

    mW, mt  =  80.4, 173.0
    rec_top_cands["xW"] = (rec_top_cands.w.jl.mass- mW)/(0.10*rec_top_cands.w.jl.mass) 
    rec_top_cands["w", "jl_wCor"] = rec_top_cands.w.jl * (mW/rec_top_cands.w.jl.mass)
    rec_top_cands["mbW"] = (rec_top_cands.bReg + rec_top_cands.w.jl_wCor).mass
    rec_top_cands["xbW"]  = (rec_top_cands.mbW-mt)/(0.05*rec_top_cands.mbW) #smaller resolution term because there are fewer degrees of freedom. FWHM=25GeV, about the same as mW 
    rec_top_cands["xWbW"] = np.sqrt( rec_top_cands.xW ** 2 + rec_top_cands.xbW**2)

    rec_top_cands = rec_top_cands[ak.argsort(rec_top_cands.xWbW, axis=1, ascending=True)]
    return rec_top_cands
