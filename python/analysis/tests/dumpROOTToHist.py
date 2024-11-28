#
# python make_weights.py -o tmp/ -c passPreSel -r SB
#
from coffea.util import save

import sys
from copy import copy
#import optparse
import argparse
import math
import numpy as np
from array import array
import os
#import collections
from coffea.util import load
import matplotlib.pyplot as plt
import hist
from scipy.optimize import curve_fit
from hist import Hist

sys.path.insert(0, os.getcwd())
from base_class.plots.plots import load_config, load_hists, read_axes_and_cuts
import base_class.plots.iPlot_config as cfg

from base_class.JCMTools import getCombinatoricWeight, getPseudoTagProbs

def data_from_TH1F(inTH1F):
    centers = []
    values  = []
    errors  = []

    for ibin in range(1, inTH1F.GetSize()-2):#data4b.GetNbinsX()-2): #GetXaxis().GetNbins()-2):
        centers.append(inTH1F.GetBinCenter (ibin))
        values .append(inTH1F.GetBinContent(ibin))
        errors .append(inTH1F.GetBinError  (ibin))
    return np.array(centers, float), np.array(values, float), np.array(errors, float)



# variables = []
def get(rootFile, path):
    obj = rootFile.Get(path)
    if str(obj) == "<ROOT.TObject object at 0x(nil)>":
        rootFile.ls()
        print()
        print( "ERROR: Object not found -", rootFile, path)
        sys.exit()

    else: return obj

# Only works for fixed size bins !
def hist_from_TH1F(inTH1F):
    centers = []
    values  = []
    errors  = []

    for ibin in range(1, inTH1F.GetSize()-1):#data4b.GetNbinsX()-2): #GetXaxis().GetNbins()-2):
        centers.append(inTH1F.GetBinCenter (ibin))
        values .append(inTH1F.GetBinContent(ibin))
        errors .append(inTH1F.GetBinError  (ibin))

    nBins = inTH1F.GetNbinsX()
    lower = inTH1F.GetXaxis().GetXmin()
    upper = inTH1F.GetXaxis().GetXmax()
    print(nBins, lower, upper, len(values), inTH1F.GetSize()-2, inTH1F.GetNbinsX())
    print(centers, values, errors)

    # Create a weighted histogram
    hist_weighted = Hist.new.Reg(nBins, lower, upper).Weight()

#    # Assume some bin contents and errors
#    bin_contents = np.zeros(nBins+2)
#    bin_contents[0] = 69
#    bin_contents[1:nBins+1] = np.array(values)
#    bin_contents[-1] = 69
#    #print(bin_contents)
#
#    bin_errors = np.zeros(nBins+2)
#    bin_errors[0] = 69
#    bin_errors[1:nBins+1] = np.array(errors)
#    bin_errors[-1] = 69

    # Set contents and variances
    hist_weighted.view().value = np.array(values)
    hist_weighted.view().variance = np.array(errors)**2

    return hist_weighted




def getHists(cut,region,var,inFile4b, inFile3b, ttFile4b, ttFile3b, plot=False):#allow for different cut for mu calculation
    baseName = cut+"_"+region+"_"+var#+("_use_mu" if mu_cut else "")
    data4b_ROOT = inFile4b.Get(cut+"/fourTag/mainView/"+region+"/"+var)

    data4b = hist_from_TH1F(data4b_ROOT)
#    print(type(data4b_hist))
#
#    import sys
#    sys.exit(-1)
#
#
#    def data_from_TH1F(inTH1F):
#
#
#    data4b.SetDirectory(0)
#    try:
#        data4b.SetName("data4b_"+baseName)
#    except:
#        inFile4b.ls()
#    data4b.Sumw2()
    data3b_ROOT = inFile3b.Get(cut+"/threeTag/mainView/"+region+"/"+var)
    data3b = hist_from_TH1F(data3b_ROOT)

#    data3b.SetName("data3b_"+baseName)
#    data3b.SetDirectory(0)

    tt4b_ROOT = ttFile4b.Get(cut+"/fourTag/mainView/"+region+"/"+var)
    tt4b = hist_from_TH1F(tt4b_ROOT)

#    tt4b.SetName("tt4b_"+baseName)
#    tt4b.SetDirectory(0)

    tt3b_ROOT = ttFile3b.Get(cut+"/threeTag/mainView/"+region+"/"+var)
    tt3b = hist_from_TH1F(tt3b_ROOT)
#    tt3b.SetName("tt3b_"+baseName)
#    tt3b.SetDirectory(0)

#    #
#    # Make qcd histograms
#    #
#    print( "str(data3b) is", str(data3b))
#    if "TH1" in str(type(data3b)):
#        qcd3b = ROOT.TH1F(data3b)
#        qcd3b.SetDirectory(0)
#        qcd3b.SetName("qcd3b_"+baseName)
#        qcd4b = ROOT.TH1F(data4b)
#        qcd4b.SetDirectory(0)
#        qcd4b.SetName("qcd4b_"+baseName)
#
#    if tt4b:
#        qcd3b.Add(tt3b,-1)
#        qcd4b.Add(tt4b,-1)
#
#    print(type(data3b))
#    print(type(tt3b))
    qcd3b = copy(data3b)
    qcd3b.view().value = data3b.values() - tt3b.values()
    qcd3b.view().variance = data3b.variances() + tt3b.variances()

    qcd4b = copy(data4b)
    qcd4b.view().value = data4b.values() - tt4b.values()
    qcd4b.view().variance = data4b.variances() + tt4b.variances()

    return (data4b, tt4b, qcd4b, data3b, tt3b, qcd3b)


def loadROOTHists():
    import ROOT
    ROOT.gROOT.SetBatch(True)

    data3bHist = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//dataRunII/hists_3b_newSBDef.root"
    data4bHist = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//dataRunII/hists_4b_newSBDef.root"
    tt3bHists  = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//TTRunII/hists_3b_newSBDef.root"
    tt4bHists  = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//TTRunII/hists_4b_newSBDef.root"

    inFile3b = ROOT.TFile.Open(data3bHist)
    inFile4b = ROOT.TFile.Open(data4bHist)
    ttFile3b = ROOT.TFile.Open(tt3bHists)
    ttFile4b = ROOT.TFile.Open(tt4bHists)


    (data4b, tt4b, qcd4b, data3b, tt3b, qcd3b)   = getHists(cut,args.weightRegion,"nSelJetsUnweighted", inFile4b, inFile3b, ttFile4b, ttFile3b)
    (data4b_nTagJets, tt4b_nTagJets, _, _, _, _) = getHists(cut,args.weightRegion,"nPSTJetsUnweighted", inFile4b, inFile3b, ttFile4b, ttFile3b)
    (_, _, _, _, _, qcd3b_nTightTags)            = getHists(cut,args.weightRegion,"nTagJetsUnweighted", inFile4b, inFile3b, ttFile4b, ttFile3b)
    #print(f"data4b is {data4b}" )
    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags



if __name__ == "__main__":


    #parser = optparse.OptionParser()
    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-c',dest="cut",default="passXWt")
    parser.add_argument('-r',dest="weightRegion",default="")
    parser.add_argument('-o', '--outputDir',dest='outputDir',default="")
    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="analysis/metadata/plotsAll.yml",
                        help='Metadata file.')

    args = parser.parse_args()
    #o, a = parser.parse_args()


    cut=args.cut


    #
    # Get Hists
    #
    data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags = loadROOTHists()

    hfile = f"{args.outputDir}"
    print(f'\nSaving file {hfile}')
    output = {}
    output["Hists"] = {}
    output["Hists"]["data4b"] =            data4b
    output["Hists"]["data3b"] =             data3b
    output["Hists"]["tt4b"] =               tt4b
    output["Hists"]["tt3b"] =               tt3b
    output["Hists"]["qcd4b"] =              qcd4b
    output["Hists"]["qcd3b"] =              qcd3b
    output["Hists"]["data4b_nTagJets"]  =   data4b_nTagJets
    output["Hists"]["tt4b_nTagJets"]    =   tt4b_nTagJets
    output["Hists"]["qcd3b_nTightTags"] =   qcd3b_nTightTags


    save(output, hfile)
