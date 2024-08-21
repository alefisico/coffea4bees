
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
