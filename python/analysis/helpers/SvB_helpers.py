import numpy as np
import awkward as ak
from base_class.math.random import Squares
import logging

def setSvBVars(SvBName, event):

    event[SvBName, "passMinPs"] = ( (getattr(event, SvBName).pzz > 0.01)
                                    | (getattr(event, SvBName).pzh > 0.01)
                                    | (getattr(event, SvBName).phh > 0.01) )

    event[SvBName, "zz"] = ( getattr(event, SvBName).pzz >  getattr(event, SvBName).pzh ) & (getattr(event, SvBName).pzz > getattr(event, SvBName).phh)

    event[SvBName, "zh"] = ( getattr(event, SvBName).pzh >  getattr(event, SvBName).pzz ) & (getattr(event, SvBName).pzh > getattr(event, SvBName).phh)

    event[SvBName, "hh"] = ( getattr(event, SvBName).phh >= getattr(event, SvBName).pzz ) & (getattr(event, SvBName).phh >= getattr(event, SvBName).pzh)

    event[SvBName, "tt_vs_mj"] = ( getattr(event, SvBName).ptt / (getattr(event, SvBName).ptt + getattr(event, SvBName).pmj) )

    #
    #  Set ps_{bb}
    #
    this_ps_zz = np.full(len(event), -1, dtype=float)
    this_ps_zz[getattr(event, SvBName).zz] = getattr(event, SvBName).ps[ getattr(event, SvBName).zz ]

    this_ps_zz[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_zz"] = this_ps_zz

    this_ps_zh = np.full(len(event), -1, dtype=float)
    this_ps_zh[getattr(event, SvBName).zh] = getattr(event, SvBName).ps[ getattr(event, SvBName).zh ]

    this_ps_zh[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_zh"] = this_ps_zh

    this_ps_hh = np.full(len(event), -1, dtype=float)
    this_ps_hh[getattr(event, SvBName).hh] = getattr(event, SvBName).ps[ getattr(event, SvBName).hh ]

    this_ps_hh[getattr(event, SvBName).passMinPs == False] = -2
    event[SvBName, "ps_hh"] = this_ps_hh


def compute_SvB(events, mask, classifier_SvB, classifier_SvB_MA, doCheck=True):

    # import torch on demand
    import torch
    import torch.nn.functional as F

    masked_events = events[mask]
    n = len(masked_events)

    j = torch.zeros(n, 4, 4)
    j[:, 0, :] = torch.tensor(masked_events.canJet.pt)
    j[:, 1, :] = torch.tensor(masked_events.canJet.eta)
    j[:, 2, :] = torch.tensor(masked_events.canJet.phi)
    j[:, 3, :] = torch.tensor(masked_events.canJet.mass)

    o = torch.zeros(n, 5, 8)
    o[:, 0, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(masked_events.notCanJet_coffea.pt,       target=8, clip=True) ), -1, ) )
    o[:, 1, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(masked_events.notCanJet_coffea.eta,      target=8, clip=True) ), -1, ) )
    o[:, 2, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(masked_events.notCanJet_coffea.phi,      target=8, clip=True) ), -1, ) )
    o[:, 3, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(masked_events.notCanJet_coffea.mass,     target=8, clip=True) ), -1, ) )
    o[:, 4, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(masked_events.notCanJet_coffea.isSelJet, target=8, clip=True) ), -1, ) )
    print(f"{events.event[0]} Computing SvB for {n} events\n\n\n\n")

    a = torch.zeros(n, 4)
    a[:, 0] = float(masked_events.metadata["year"][3])
    a[:, 1] = torch.tensor(masked_events.nJet_selected)
    a[:, 2] = torch.tensor(masked_events.xW)
    a[:, 3] = torch.tensor(masked_events.xbW)

    e = torch.tensor(masked_events.event) % 3

    for classifier in ["SvB", "SvB_MA"]:

        if classifier in events.fields: 
            events[f"old_{classifier}"] = events[classifier]
        
        if classifier == "SvB":
            print(f"{events.event[0]} shape of j: {j.shape}, n = {n}")
            c_logits, q_logits = classifier_SvB(j, o, a, e)

        if classifier == "SvB_MA":
            c_logits, q_logits = classifier_SvB_MA(j, o, a, e)

        tmp_c_score, tmp_q_score = ( F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy(), )
        c_score = np.zeros((events.__len__(),5))
        c_score[mask] = tmp_c_score
        q_score = np.zeros((events.__len__(),3))
        q_score[mask] = tmp_q_score

        # classes = [mj,tt,zz,zh,hh]
        pmj = c_score[:, 0]
        ptt = c_score[:, 1]
        pzz = c_score[:, 2]
        pzh = c_score[:, 3]
        phh = c_score[:, 4]
        ps = pzz + pzh + phh
        passMinPs = (pzz > 0.01) | (pzh > 0.01) | (phh > 0.01)

        zz = (pzz > pzh) & (pzz > phh)
        this_ps_zz = np.full(len(events), -1, dtype=float)
        this_ps_zz[ zz ] = ps[zz]
        this_ps_zz[ passMinPs == False ] = -2
        ps_zz = this_ps_zz

        zh = (pzh > pzz) & (pzh > phh)  
        this_ps_zh = np.full(len(events), -1, dtype=float)
        this_ps_zh[ zh ] = ps[zh]
        this_ps_zh[ passMinPs == False ] = -2
        ps_zh = this_ps_zh

        hh = (phh > pzz) & (phh > pzh)  
        this_ps_hh = np.full(len(events), -1, dtype=float)
        this_ps_hh[ hh ] = ps[hh]
        this_ps_hh[ passMinPs == False ] = -2
        ps_hh = this_ps_hh

        largest_name = np.array(["None", "ZZ", "ZH", "HH"])
        events[classifier] = ak.zip({
            "pmj": pmj,
            "ptt": ptt,
            "pzz": pzz,
            "pzh": pzh,
            "phh": phh,
            "q_1234": q_score[:, 0],
            "q_1324": q_score[:, 1],
            "q_1423": q_score[:, 2],
            "ps": ps,
            "passMinPs": passMinPs,
            "zz": zz,
            "zh": zh,
            "hh": hh,
            "ps_zz": ps_zz,
            "ps_zh": ps_zh,
            "ps_hh": ps_hh,
            "largest": largest_name[ (passMinPs * ( 1 * zz + 2* zh + 3*hh ) ) ],
            "tt_vs_mj": ( ptt / (ptt + pmj) )
        })

        if doCheck and f"old_{classifier}" in events.fields:
            error = ~np.isclose(events[f"old_{classifier}"].ps, events[classifier].ps, atol=1e-5, rtol=1e-3)
            if np.any(error):
                delta = np.abs(events[f"old_{classifier}"].ps - events[classifier].ps)
                worst = np.max(delta) == delta
                worst_events = events[worst][0]

                logging.warning( f"WARNING: Calculated {classifier} does not agree within tolerance for some events ({np.sum(error)}/{len(error)}) {delta[worst]}" )

                logging.warning("----------")

                for field in events[classifier].fields:
                    logging.warning(f"{field} {worst_events[classifier][field]}")

                logging.warning("----------")

                for field in events[classifier].fields:
                    logging.warning(f"{field} {events[classifier][worst][field]}")



def subtract_ttbar_with_SvB(selev, dataset, year):

    #
    # Get reproducible random numbers
    #
    rng = Squares("ttbar_subtraction", dataset, year)
    counter = np.empty((len(selev), 2), dtype=np.uint64)
    counter[:, 0] = np.asarray(selev.event).view(np.uint64)
    counter[:, 1] = np.asarray(selev.run).view(np.uint32)
    counter[:, 1] <<= 32
    counter[:, 1] |= np.asarray(selev.luminosityBlock).view(np.uint32)
    ttbar_rand = rng.uniform(counter, low=0, high=1.0).astype(np.float32)

    return (ttbar_rand > selev.SvB_MA.tt_vs_mj)
