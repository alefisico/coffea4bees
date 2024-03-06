from scipy.special import comb
import numpy as np
from coffea.util import load
from base_class.plots.plots import load_config, load_hists, read_axes_and_cuts, get_cut_dict
import base_class.plots.iPlot_config as cfg
import hist
from copy import copy

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


def loadROOTHists(inputFile):

    #
    #  > From python analysis/tests/dumpROOTToHist.py -o analysis/tests/HistsFromROOTFile.coffea -c passPreSel -r SB 
    #
    h = load(inputFile)["Hists"]

    data4b           = h["data4b"]
    data3b           = h["data3b"]
    tt4b             = h["tt4b"]
    tt3b             = h["tt3b"]
    qcd4b            = h["qcd4b"]
    qcd3b            = h["qcd3b"]
    data4b_nTagJets  = h["data4b_nTagJets"]
    tt4b_nTagJets    = h["tt4b_nTagJets"]
    qcd3b_nTightTags = h["qcd3b_nTightTags"]

    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags


def loadCoffeaHists(inputFile, metadata, *, cut="passPreSel", year="RunII", weightRegion="SB"):

    cfg.plotConfig = load_config(metadata)
    cfg.hists = load_hists([inputFile])
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    cutDict = get_cut_dict(cut, cfg.cutList)

    codes = cfg.plotConfig["codes"]
    year = sum if year == "RunII" else year
    region_selection = sum if weightRegion in ["sum", sum] else hist.loc(codes["region"][weightRegion])

    region_year_dict = {
        "year":    year,
        "region":  region_selection,
    }

    fourTag_dict  = {"tag":hist.loc(codes["tag"]["fourTag"])}
    threeTag_dict = {"tag":hist.loc(codes["tag"]["threeTag"])}

    fourTag_data_dict  = {"process":'data'} | fourTag_dict | region_year_dict | cutDict
    threeTag_data_dict = {"process":'data'} | threeTag_dict | region_year_dict | cutDict

    ttbar_list = ['TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic']
    fourTag_ttbar_dict   = {"process":ttbar_list} | fourTag_dict  | region_year_dict | cutDict
    threeTag_ttbar_dict  = {"process":ttbar_list} | threeTag_dict | region_year_dict | cutDict

    hists = cfg.hists[0]['hists']

    data4b                = hists['selJets_noJCM.n']      [fourTag_data_dict]
    data4b_nTagJets       = hists['tagJets_loose_noJCM.n'][fourTag_data_dict]

    data3b                = hists['selJets_noJCM.n']      [threeTag_data_dict]
    data3b_nTagJets       = hists['tagJets_loose_noJCM.n'][threeTag_data_dict]
    data3b_nTagJets_tight = hists['tagJets_noJCM.n']      [threeTag_data_dict]

    tt4b                  = hists['selJets_noJCM.n']      [fourTag_ttbar_dict][sum,:]
    tt4b_nTagJets         = hists['tagJets_loose_noJCM.n'][fourTag_ttbar_dict][sum,:]
                               
    tt3b                  = hists['selJets_noJCM.n']      [threeTag_ttbar_dict][sum,:]
    tt3b_nTagJets         = hists['tagJets_loose_noJCM.n'][threeTag_ttbar_dict][sum,:]
    tt3b_nTagJets_tight   = hists['tagJets_noJCM.n']      [threeTag_ttbar_dict][sum,:]

    qcd4b = copy(data4b)
    qcd4b.view().value = data4b.values() - tt4b.values()
    qcd4b.view().variance = data4b.variances() + tt4b.variances()

    qcd3b = copy(data3b)
    qcd3b.view().value = data3b.values() - tt3b.values()
    qcd3b.view().variance = data3b.variances() + tt3b.variances()

    qcd3b_nTightTags = copy(data3b_nTagJets_tight)
    qcd3b_nTightTags.view().value = data3b_nTagJets_tight.values() - tt3b_nTagJets_tight.values()
    qcd3b_nTightTags.view().variance = data3b_nTagJets_tight.variances() + tt3b_nTagJets_tight.variances()
    
    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags

def data_from_Hist(inputHist, maxBin=15):
    x_centers = inputHist.axes[0].centers
    values = inputHist.values()
    errors = np.sqrt( inputHist.variances() )
    
    if x_centers[0] == 0.5:
        x_centers = x_centers - 0.5

    return x_centers[0:maxBin], values[0:maxBin], errors[0:maxBin]
