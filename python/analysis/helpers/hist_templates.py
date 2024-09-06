from base_class.physics.object import LorentzVector, Jet
from base_class.hist import H, Template

class SvBHists(Template):
    ps      = H((50, 0, 1, ('ps', "Regressed P(Signal)")))
    ptt     = H((50, 0, 1, ('ptt', "Regressed P(tT)")))

    tt_vs_mj     = H((50, 0, 1, ('tt_vs_mj', "P(tT) | Background")))

    ps_zz   = H((200, 0, 1, ('ps_zz', "Regressed P(Signal) $|$ P(ZZ) is largest ")))
    ps_zh   = H((200, 0, 1, ('ps_zh', "Regressed P(Signal) $|$ P(ZH) is largest ")))
    ps_hh   = H((200, 0, 1, ('ps_hh', "Regressed P(Signal) $|$ P(HH) is largest ")))

    ps_zz_fine   = H((240, 0, 1, ('ps_zz', "Regressed P(Signal) $|$ P(ZZ) is largest ")))
    ps_zh_fine   = H((240, 0, 1, ('ps_zh', "Regressed P(Signal) $|$ P(ZH) is largest ")))
    ps_hh_fine   = H((240, 0, 1, ('ps_hh', "Regressed P(Signal) $|$ P(HH) is largest ")))


class FvTHists(Template):
    FvT  = H((50, 0, 5, ('FvT', 'FvT reweight')))
    pd4  = H((50, 0, 1, ("pd4",   'FvT Regressed P(Four-tag Data)')))
    pd3  = H((50, 0, 1, ("pd3",   'FvT Regressed P(Three-tag Data)')))
    pt4  = H((50, 0, 1, ("pt4",   'FvT Regressed P(Four-tag t#bar{t})')))
    pt3  = H((50, 0, 1, ("pt3",   'FvT Regressed P(Three-tag t#bar{t})')))
    pm4  = H((50, 0, 1, ("pm4",   'FvT Regressed P(Four-tag Multijet)')))
    pm3  = H((50, 0, 1, ("pm3",   'FvT Regressed P(Three-tag Multijet)')))
    pt   = H((50, 0, 1, ("pt",    'FvT Regressed P(t#bar{t})')))
    std  = H((50, 0, 3, ("std",   'FvT Standard Deviation')))
    frac_err = H((50, 0, 5, ("frac_err",  'FvT std/FvT')))
    #'q_1234', 'q_1324', 'q_1423',

class QuadJetHists(Template):
    dr              = H((50,     0, 5,   ("dr",          'Diboson Candidate $\\Delta$R(d,d)')))
    dphi            = H((50, -3.2, 3.2, ("dphi",        'Diboson Candidate $\\Delta$R(d,d)')))
    deta            = H((50,   -5, 5,   ("deta",        'Diboson Candidate $\\Delta$R(d,d)')))
    xZZ             = H((50, 0, 10,     ("xZZ",         'Diboson Candidate zZZ')))
    xZH             = H((50, 0, 10,     ("xZH",         'Diboson Candidate zZH')))
    xHH             = H((50, 0, 10,     ("xHH",         'Diboson Candidate zHH')))

    lead_vs_subl_m   = H((50, 0, 250, ('lead.mass', 'Lead Boson Candidate Mass')),
                         (50, 0, 250, ('subl.mass', 'Subl Boson Candidate Mass')))

    close_vs_other_m = H((50, 0, 250, ('close.mass', 'Close Boson Candidate Mass')),
                         (50, 0, 250, ('other.mass', 'Other Boson Candidate Mass')))

    lead            = LorentzVector.plot_pair(('...', R'Lead Boson Candidate'),  'lead',  skip=['n'])
    subl            = LorentzVector.plot_pair(('...', R'Subl Boson Candidate'),  'subl',  skip=['n'])
    close           = LorentzVector.plot_pair(('...', R'Close Boson Candidate'), 'close', skip=['n'])
    other           = LorentzVector.plot_pair(('...', R'Other Boson Candidate'), 'other', skip=['n'])

class QuadJetHistsUnsup(Template):
    dr              = H((50,     0, 5,   ("dr",          'Diboson Candidate $\\Delta$R(d,d)')))
    dphi            = H((100, -3.2, 3.2, ("dphi",        'Diboson Candidate $\\Delta$R(d,d)')))
    deta            = H((100,   -5, 5,   ("deta",        'Diboson Candidate $\\Delta$R(d,d)')))

    lead_vs_subl_m   = H((50, 0, 250, ('lead.mass', 'Lead Boson Candidate Mass')),
                         (50, 0, 250, ('subl.mass', 'Subl Boson Candidate Mass')))

    close_vs_other_m = H((50, 0, 250, ('close.mass', 'Close Boson Candidate Mass')),
                         (50, 0, 250, ('other.mass', 'Other Boson Candidate Mass')))

    lead            = LorentzVector.plot_pair(('...', R'Lead Boson Candidate'),  'lead',  skip=['n'])
    subl            = LorentzVector.plot_pair(('...', R'Subl Boson Candidate'),  'subl',  skip=['n'])
    close           = LorentzVector.plot_pair(('...', R'Close Boson Candidate'), 'close', skip=['n'])
    other           = LorentzVector.plot_pair(('...', R'Other Boson Candidate'), 'other', skip=['n'])


class WCandHists(Template):

    p  = LorentzVector.plot(('...', R'W Candidate'), 'p',  skip=['n'])
    pW = LorentzVector.plot(('...', R'W Candidate'), 'pW', skip=['n'])

    j = Jet.plot(('...', R'W j jet Candidate'), 'j',     skip=['deepjet_c','n'])
    l = Jet.plot(('...', R'W l jet Candidate'), 'l',     skip=['deepjet_c','n'])

class TopCandHists(Template):

    t = LorentzVector.plot(('...', R'Top Candidate'), 'p', skip=['n'])
    b = Jet.plot(('...', R'Top b jet Candidate'), 'b', skip=['deepjet_c','n'])
    W = WCandHists(('...', R'W boson Candidate'), 'W')

    mbW  = H(( 50, 80, 280,   ("mbW",  'm_{b,W}')))
    xWt  = H(( 24, 0,  6,   ("xWt",  "X_{W,t}")))
    xWbW = H(( 24, 0,  6,   ("xWbW", "X_{W,bW}")))
    rWbW = H(( 24, 0,  6,   ("rWbW", "r_{W,bW}")))
    xbW  = H(( 60, -15,  15,   ("xbW",  "X_{W,bW}")))
    xW   = H(( 24, 0,  6,   ("xW",   'X_{W}')))

    mW_vs_mt  = H((50,  0, 250, ('W.p.mass', 'W Candidate Mass [GeV]')),
                  (50, 80, 280, ('p.mass',   'Top Candidate Mass [GeV]')))

    mW_vs_mbW = H((50,  0, 250, ('W.p.mass', 'W Candidate Mass [GeV]')),
                  (50, 80, 280, ('mbW',   'm_{b,W} [GeV]')))

    xW_vs_xt  = H((24, 0,  6,   ("xW",   'X_{W}')),
                  (24, 0,  6,   ("xt",   'X_{t}')))

    xW_vs_xbW  = H((24, 0,  6,   ("xW",   'X_{W}')),
                   (24, 0,  6,   ("xbW",  'X_{bW}')))
