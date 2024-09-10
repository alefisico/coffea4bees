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


def compute_SvB(event, classifier_SvB, classifier_SvB_MA, doCheck=True):
    # import torch on demand
    import torch
    import torch.nn.functional as F

    n = len(event)

    j = torch.zeros(n, 4, 4)
    j[:, 0, :] = torch.tensor(event.canJet.pt)
    j[:, 1, :] = torch.tensor(event.canJet.eta)
    j[:, 2, :] = torch.tensor(event.canJet.phi)
    j[:, 3, :] = torch.tensor(event.canJet.mass)

    o = torch.zeros(n, 5, 8)
    o[:, 0, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.pt,       target=8, clip=True) ), -1, ) )
    o[:, 1, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.eta,      target=8, clip=True) ), -1, ) )
    o[:, 2, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.phi,      target=8, clip=True) ), -1, ) )
    o[:, 3, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.mass,     target=8, clip=True) ), -1, ) )
    o[:, 4, :] = torch.tensor( ak.fill_none( ak.to_regular( ak.pad_none(event.notCanJet_coffea.isSelJet, target=8, clip=True) ), -1, ) )

    a = torch.zeros(n, 4)
    a[:, 0] = float(event.metadata["year"][3])
    a[:, 1] = torch.tensor(event.nJet_selected)
    a[:, 2] = torch.tensor(event.xW)
    a[:, 3] = torch.tensor(event.xbW)

    e = torch.tensor(event.event) % 3

    for classifier in ["SvB", "SvB_MA"]:

        if classifier == "SvB":
            c_logits, q_logits = classifier_SvB(j, o, a, e)

        if classifier == "SvB_MA":
            c_logits, q_logits = classifier_SvB_MA(j, o, a, e)

        c_score, q_score = ( F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy(), )

        # classes = [mj,tt,zz,zh,hh]
        SvB = ak.zip( { "pmj": c_score[:, 0],
                        "ptt": c_score[:, 1],
                        "pzz": c_score[:, 2],
                        "pzh": c_score[:, 3],
                        "phh": c_score[:, 4],
                        "q_1234": q_score[:, 0],
                        "q_1324": q_score[:, 1],
                        "q_1423": q_score[:, 2],
                        }
                        )

        largest_name = np.array(["None", "ZZ", "ZH", "HH"])
        SvB["ps"] = SvB.pzz + SvB.pzh + SvB.phh
        SvB["passMinPs"] = (SvB.pzz > 0.01) | (SvB.pzh > 0.01) | (SvB.phh > 0.01)
        SvB["zz"] = (SvB.pzz > SvB.pzh) & (SvB.pzz > SvB.phh)
        SvB["zh"] = (SvB.pzh > SvB.pzz) & (SvB.pzh > SvB.phh)
        SvB["hh"] = (SvB.phh > SvB.pzz) & (SvB.phh > SvB.pzh)
        SvB["largest"] = largest_name[ SvB.passMinPs * ( 1 * SvB.zz + 2* SvB.zh + 3*SvB.hh ) ]

        this_ps_zz = np.full(len(event), -1, dtype=float)
        this_ps_zz[ SvB.zz ] = SvB.ps[ SvB.zz ]
        this_ps_zz[ SvB.passMinPs == False ] = -2
        SvB['ps_zz'] = this_ps_zz

        this_ps_zh = np.full(len(event), -1, dtype=float)
        this_ps_zh[ SvB.zh ] = SvB.ps[ SvB.zh ]
        this_ps_zh[ SvB.passMinPs == False ] = -2
        SvB['ps_zh'] = this_ps_zh

        this_ps_hh = np.full(len(event), -1, dtype=float)
        this_ps_hh[ SvB.hh ] = SvB.ps[ SvB.hh ]
        this_ps_hh[ SvB.passMinPs == False ] = -2
        SvB['ps_hh'] = this_ps_hh

        SvB['tt_vs_mj'] = ( SvB.ptt / (SvB.ptt + SvB.pmj) )

        if doCheck and classifier in event.fields:
            error = ~np.isclose(event[classifier].ps, SvB.ps, atol=1e-5, rtol=1e-3)
            if np.any(error):
                delta = np.abs(event[classifier].ps - SvB.ps)
                worst = np.max(delta) == delta
                worst_event = event[worst][0]

                logging.warning( f"WARNING: Calculated {classifier} does not agree within tolerance for some events ({np.sum(error)}/{len(error)}) {delta[worst]}" )

                logging.warning("----------")

                for field in event[classifier].fields:
                    logging.warning(f"{field} {worst_event[classifier][field]}")

                logging.warning("----------")

                for field in SvB.fields:
                    logging.warning(f"{field} {SvB[worst][field]}")

        # del event[classifier]
        event[classifier] = SvB


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
