#
# python make_weights.py -o tmp/ -c passPreSel -r SB
#
import ROOT
ROOT.gROOT.SetBatch(True)
import operator as op
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
from functools import reduce
from past.builtins import xrange
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from analysis.iPlot import load_config, load_hists, read_axes_and_cuts
import analysis.iPlot_config as cfg

def ncr(n, r):
    r = min(r, n-r)
    if r <= 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom #double slash means integer division or "floor" division

def nPairings(n):
    pairings=1
    if n<=1: return 0
    if n%2:
        pairings = n #n options for the item to be unpaired
        n = n-1 # now need so solve the simpler case with n even
    nPairs = n//2
    pairings *= reduce(op.mul, xrange(n, nPairs, -1))//(2**nPairs)
    return pairings


# def getCombinatoricWeight(f,nj,pairEnhancement=0.0,pairEnhancementDecay=1.0, unTaggedPartnerRate=0.0, pairRate=0.0, singleRate=1.0):
#     w = 0
#     nbt = 3 #number of required bTags
#     nlt = nj-nbt #number of selected untagged jets ("light" jets)
#     nPseudoTagProb = []
#     unPseudoTagProb = 1-singleRate-pairRate-unTaggedPartnerRate
#     for npt in range(0,nlt + 1):#npt is the number of pseudoTags in this combination
#         ntg = nbt+npt
#         nut = nlt-npt

#         w_npt = 0
#         # nested loop over all possible configurations of pseudoTags
#         for npr in range(0,npt+1): # loop over all possible number of pseudoTags which were pair produced
#             # (all ways to choose npt pseudotags from nlt light tagged jets) * (all ways to choose npr pseudotags which were pair produced from npt pseudotags) * pairRate^npr
#             p_npr = ncr(nlt,npt) * ncr(npt,npr) * pairRate**npr
#             w_npt_npr = 0

#             #for nup in range(0,npt-npr+1): # loop over all possible untagged partners to true b pseudotags
#             for nup in range(0,min(nut,npt-npr)+1): # loop over all possible untagged partners to true b pseudotags
#                 for nsg in range(0,npt-npr-nup+1): # loop over all possible pseudotags where pair produced b did not end up in preselected jet collection
#                     p_nsg = ncr(npt-npr-nup,nsg) * singleRate**nsg # prob to get nsg-nup single pseudo-tags where the partner b-jet was out of acceptance
#                     # Fakes and pair produced b's out of acceptance look the same and can be encapsulated in one parameter called "singleRate"
#                     # The case where the pair produced b was untagged needs an additional parameter due to the fact that it is a true b which was untagged.

#                     # (all ways to choose nup from remaining pseudotags) (all ways to choose nup unTagged partners to true b pseudotags from nut untagged jets) * pairRate^nup * unTaggedPartnerRate^nup * unPseudoTagProb^(nut-nup)
#                     p_nup = ncr(nut,nup) * pairRate**nup * unTaggedPartnerRate**nup * unPseudoTagProb**(nut-nup)
#                     p_nup *=

#                     # all ways to choose nsg single pseudotags where the pair produced b-jet did not pass jet preselection from the available npt-npr pseudotags * singleRate^(nsg-)
#                     # nsg = npt-npr-nup-nupb
#                     w_npt_npr_nup = p_npr * p_nup * p_nsg
#                     w_npt_npr += w_npt_npr_nup
#                     print('npt, npr, nup, nupb, nsg',npt,npr,nup,nupb,nsg, 'w_npt_npr_nup =',w_npt_npr_nup,'= p_npr * p_nup * p_nsg =',p_npr,'*',p_nup,'*',p_nsg

#                     w_npt += w_npt_npr

#         #  (ways to choose i pseudoTags from nlt light jets) * pseudoTagProb^i * (1-pseudoTagProb)^{nut}
#         # w_npt = ncr(nlt,i) * f**i * (1-f)**(nut)
#         # if(i%2)==1: w_npt *= 1 + pairEnhancement/(nlt**pairEnhancementDecay)

#         nPseudoTagProb.append(w_npt)
#         if npt>=4-nbt: w += w_npt

#     if abs(singleRate-0.0775166688549)<1e-3 or True:
#         print(sum(nPseudoTagProb), nPseudoTagProb
#         raw_input()

#     return w, nPseudoTagProb


# def getCombinatoricWeight(f,nj,pairEnhancement=0.0,pairEnhancementDecay=1.0, unTaggedPartnerRate=0.0, pairRate=0.0, singleRate=1.0, fakeRate = 0.0):
#     w = 0
#     nbt = 3 #number of required bTags
#     nlt = nj-nbt #number of selected untagged jets ("light" jets)
#     nTagProb = np.zeros(nj+1)
#     nSingleTagProb = np.zeros(nj+1)

#     max_npr = nj//2
#     for npr in range(0,max_npr+1):#npr is the number of tag jet pairs
#         p_npr = ncr(max_npr,npr) * (pairRate)**npr #* (1-pairRate)**(max_npr-npr)
#         max_nup = max_npr-npr
#         for nup in range(0,max_nup+1):#nup is the number of b-jet pairs where only one was tagged but both were kinematically selected
#             p_nup = ncr(max_nup,nup) * unTaggedPartnerRate**nup * (1-unTaggedPartnerRate-pairRate)**(max_npr-npr-nup)
#             max_nsg = nj-2*(npr+nup)
#             for nsg in range(0,max_nsg+1):#nsg is the number of single tags
#                 if npr or nup or nsg<3: continue
#                 p_nsg = ncr(max_nsg,nsg) * singleRate**nsg * (1-singleRate)**(max_nsg-nsg)
#                 nt = 2*npr+nup+nsg
#                 nTagProb[nt] += p_npr * p_nup * p_nsg

#     nPseudoTagProb = nTagProb[3:]/np.sum(nTagProb[3:])
#     w = sum(nPseudoTagProb[1:])
#     return w, nPseudoTagProb


def getCombinatoricWeight(nj, f, e=0.0, d=1.0, norm=1.0):
    w = 0
    nbt = 3 #number of required bTags
    nlt = nj-nbt #number of selected untagged jets ("light" jets)
    nPseudoTagProb = np.zeros(nlt+1)
    for npt in range(0,nlt + 1):#npt is the number of pseudoTags in this combination
        nt = nbt + npt
        nnt = nlt-npt # number of not tagged
        # (ways to choose npt pseudoTags from nlt light jets) * pseudoTagProb^nlt * (1-pseudoTagProb)^{nlt-npt}
        w_npt = norm * ncr(nlt,npt) * f**npt * (1-f)**nnt
        if (nt%2) == 0: w_npt *= 1 + e/nlt**d

        nPseudoTagProb[npt] += w_npt
    w = np.sum(nPseudoTagProb[1:])
    return w, nPseudoTagProb



#parser = optparse.OptionParser()
parser = argparse.ArgumentParser(description='uproot_plots')
parser.add_argument('--noFitWeight',dest='noFitWeight',default="")
parser.add_argument('-w', '--weightSet',dest="weightSet",default="")
parser.add_argument('-r',dest="weightRegion",default="")
parser.add_argument('-c',dest="cut",default="passXWt")
parser.add_argument('-o', '--outputDir',dest='outputDir',default="")
parser.add_argument('-y', '--year',                 dest="year",          default="2018", help="Year specifies trigger (and lumiMask for data)")
parser.add_argument('-l', '--lumi',                 dest="lumi",          default="1",    help="Luminosity for MC normalization: units [pb]")
parser.add_argument('-m', '--metadata', dest="metadata",
                    default="analysis/metadata/plotsAll.yml",
                    help='Metadata file.')

args = parser.parse_args()
#o, a = parser.parse_args()

lumi = float(args.lumi)/1000

if not os.path.isdir(args.outputDir):
    os.mkdir(args.outputDir)



class modelParameter:
    def __init__(self, name="", index=0, lowerLimit=0, upperLimit=1, default=0.5, fix=None):
        self.name = name
        self.value = None
        self.error = None
        self.percentError = None
        self.index = index
        self.lowerLimit = lowerLimit
        self.upperLimit = upperLimit
        self.default = default
        self.fix = fix

    def dump(self):
        self.percentError = self.error/self.value*100 if self.value else 0
        print((self.name+" %1.4f +/- %0.5f (%1.1f%%)")%(self.value,self.error,self.percentError))

class jetCombinatoricModel:
    def __init__(self):
        self.pseudoTagProb       = modelParameter("pseudoTagProb",        index=0, lowerLimit=0,   upperLimit= 1, default=0.05)
        self.pairEnhancement     = modelParameter("pairEnhancement",      index=1, lowerLimit=0,   upperLimit= 3, default=1.0,
                                                  #fix=0,
                                                  )
        self.pairEnhancementDecay= modelParameter("pairEnhancementDecay", index=2, lowerLimit=0.1, upperLimit=100, default=0.7,
                                                  #fix=1,
                                                  )
        self.threeTightTagFraction = modelParameter("threeTightTagFraction",   index=3, lowerLimit=0, upperLimit=1000000, default=1000)

        self.parameters = [self.pseudoTagProb, self.pairEnhancement, self.pairEnhancementDecay, self.threeTightTagFraction]
        self.nParameters = len(self.parameters)

    def dump(self):
        for parameter in self.parameters:
            parameter.dump()

jetCombinatoricModelNext = args.outputDir+"jetCombinatoricModel_"+args.weightRegion+"_"+args.weightSet+".txt"
print(jetCombinatoricModelNext)
jetCombinatoricModelFile =             open(jetCombinatoricModelNext, "w")
JCMROOTFileName = jetCombinatoricModelNext.replace(".txt",".root")
jetCombinatoricModelRoot = ROOT.TFile(JCMROOTFileName,"RECREATE")
print(jetCombinatoricModelRoot, JCMROOTFileName)
jetCombinatoricModels = {}

cut=args.cut


def make_TH1F_from_Hist(inputHist, name, title):
    x_centers = inputHist.axes[0].centers
    x_edges   = inputHist.axes[0].edges

    output = ROOT.TH1F( name, title , len(x_edges)-1, x_edges[0], x_edges[-1] )
    output.Sumw2()
    values = inputHist.values()
    errors = np.sqrt( inputHist.variances() )

    #print(values)

    for i in range(1, output.GetNbinsX()):
        output.SetBinContent( i, values[i-1] )
        output.SetBinError( i, errors[i-1] )

    return output


def loadCoffeaHists():
    #infile = load("hists/histAll_testJCM.coffea")['hists']
    #infile = load("analysis/hists/histAll.coffea")['hists']
    infile = load("analysis/hists/test.coffea")['hists']

    cfg.plotConfig = load_config(args.metadata)
    cfg.hists = load_hists(["analysis/hists/test.coffea"])
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)


    cutDict = {}
    for c in cfg.cutList:
        cutDict[c] = sum
    cutDict[cut] = True

    codes = cfg.plotConfig["codes"]
    year = sum if args.year == "RunII" else args.year
    #tag = plotConfig["codes"]["tag"][tagName]
    region_selection = sum if args.weightRegion in ["sum", sum] else hist.loc(codes["region"][args.weightRegion])

    hist_dict_4tag = {
        "year":    year,
        "tag":     hist.loc(codes["tag"]["fourTag"]),
        "region":  region_selection,
    }

    data4b_hist = infile['selJets_noJCM.n']['data',sum,1,1,True,False,False,:]
    data4b = make_TH1F_from_Hist(data4b_hist, "data4b", "dat4b")

    data4b_nTagJets_hist = infile['tagJets_loose_noJCM.n']['data',sum,1,1,True,False,False,:]
    data4b_nTagJets = make_TH1F_from_Hist(data4b_nTagJets_hist, "data4b_nTagJets", "data4b_nTagJets")

    data3b_hist = infile['selJets_noJCM.n']['data',sum,0,1,True,False,False,:]
    data3b = make_TH1F_from_Hist(data3b_hist, "data3b", "dat3b")

    data3b_nTagJets_hist = infile['tagJets_loose_noJCM.n']['data',sum,0,1,True,False,False,:]
    data3b_nTagJets = make_TH1F_from_Hist(data3b_nTagJets_hist, "data3b_nTagJets", "data3b_nTagJets")

    data3b_nTagJets_tight_hist = infile['tagJets_noJCM.n']['data',sum,0,1,True,False,False,:]
    data3b_nTagJets_tight = make_TH1F_from_Hist(data3b_nTagJets_tight_hist, "data3b_nTagJets_tight", "data3b_nTagJets_tight")

    ttbar_list = ['TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic']

    tt4b_hist = infile['selJets_noJCM.n'][ttbar_list,sum,1,1,True,False,False,:][sum,:]
    tt4b = make_TH1F_from_Hist(tt4b_hist, "tt4b", "tt4b")

    tt4b_nTagJets_hist = infile['tagJets_loose_noJCM.n'][ttbar_list,sum,1,1,True,False,False,:][sum,:]
    tt4b_nTagJets = make_TH1F_from_Hist(tt4b_nTagJets_hist, "tt4b_nTagJets", "tt4b_nTagJets")

    tt3b_hist = infile['selJets_noJCM.n'][ttbar_list,sum,0,1,True,False,False,:][sum,:]
    tt3b = make_TH1F_from_Hist(tt3b_hist, "tt3b", "tt3b")

    tt3b_nTagJets_hist = infile['tagJets_loose_noJCM.n'][ttbar_list,sum,0,1,True,False,False,:][sum,:]
    tt3b_nTagJets = make_TH1F_from_Hist(tt3b_nTagJets_hist, "tt3b_nTagJets", "tt3b_nTagJets")

    tt3b_nTagJets_tight_hist = infile['tagJets_noJCM.n'][ttbar_list,sum,0,1,True,False,False,:][sum,:]
    tt3b_nTagJets_tight = make_TH1F_from_Hist(tt3b_nTagJets_tight_hist, "tt3b_nTagJets_tight", "tt3b_nTagJets_tight")
    
    qcd4b = data4b.Clone("qcd4b")
    qcd4b.Add( tt4b, -1 )
    qcd3b = data3b.Clone('qcd3b')
    qcd3b.Add( tt3b, -1 )
    #qcd4b_nTagJets = data4b_nTagJets.Clone("qcd4b")
    #qcd4b_nTagJets.Add( tt4b_nTagJets, -1 )
    qcd3b_nTightTags = data3b_nTagJets_tight.Clone('qcd3b')
    qcd3b_nTightTags.Add( tt3b_nTagJets_tight, -1 )
    
    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags


# variables = []
def get(rootFile, path):
    obj = rootFile.Get(path)
    if str(obj) == "<ROOT.TObject object at 0x(nil)>": 
        rootFile.ls()
        print() 
        print( "ERROR: Object not found -", rootFile, path)
        sys.exit()

    else: return obj


def getHists(cut,region,var,inFile4b, inFile3b, ttFile4b, ttFile3b, plot=False):#allow for different cut for mu calculation
    baseName = cut+"_"+region+"_"+var#+("_use_mu" if mu_cut else "")
    data4b = inFile4b.Get(cut+"/fourTag/mainView/"+region+"/"+var)
    data4b.SetDirectory(0)
    try:
        data4b.SetName("data4b_"+baseName)
    except:
        inFile4b.ls()
    data4b.Sumw2()
    data3b = inFile3b.Get(cut+"/threeTag/mainView/"+region+"/"+var)
    data3b.SetName("data3b_"+baseName)
    data3b.SetDirectory(0)

    tt4b = ttFile4b.Get(cut+"/fourTag/mainView/"+region+"/"+var)
    tt4b.SetName("tt4b_"+baseName)
    tt4b.SetDirectory(0)

    tt3b = ttFile3b.Get(cut+"/threeTag/mainView/"+region+"/"+var)
    tt3b.SetName("tt3b_"+baseName)
    tt3b.SetDirectory(0)

    #
    # Make qcd histograms
    #
    print( "str(data3b) is", str(data3b))
    if "TH1" in str(type(data3b)):
        qcd3b = ROOT.TH1F(data3b)
        qcd3b.SetDirectory(0)
        qcd3b.SetName("qcd3b_"+baseName)
        qcd4b = ROOT.TH1F(data4b)
        qcd4b.SetDirectory(0)
        qcd4b.SetName("qcd4b_"+baseName)

    if tt4b:
        qcd3b.Add(tt3b,-1)
        qcd4b.Add(tt4b,-1)

    if "TH1" in str(type(data3b)):
        bkgd = ROOT.TH1F(qcd3b)
        bkgd.SetName("bkgd_"+baseName)
    elif "TH2" in str(type(data3b)):
        bkgd = ROOT.TH2F(qcd3b)
        bkgd.SetName("bkgd_"+baseName)
    if tt4b:
        bkgd.Add(tt4b)

    data4b.SetLineColor(ROOT.kBlack)
    qcd3b.SetFillColor(ROOT.kYellow)
    qcd3b.SetLineColor(ROOT.kBlack)
    if tt4b:
        tt4b.SetLineColor(ROOT.kBlack)
        tt4b.SetFillColor(ROOT.kAzure-9)
        
    if plot:
        if '/' in var: var=var.replace('/','_')
        c=ROOT.TCanvas(var+"_"+cut+"_4b","")
        data4b.Draw("P EX0")
        stack = ROOT.THStack("stack","stack")
        if tt4b:
            stack.Add(tt4b,"hist")
        stack.Add(qcd3b,"hist")
        stack.Draw("HIST SAME")
        data4b.SetStats(0)
        data4b.SetMarkerStyle(20)
        data4b.SetMarkerSize(0.7)
        data4b.Draw("P EX0 SAME axis")
        data4b.Draw("P EX0 SAME")
        plotName = args.outputDir+"/"+var+"_"+cut+"_4b.pdf" 
        print( plotName)
        c.SetLogy(True)
        c.SaveAs(plotName)
        del stack

        c=ROOT.TCanvas(var+"_"+cut+"_3b","")
        data3b.SetLineColor(ROOT.kBlack)
        data3b.Draw("P EX0")
        if tt3b:
            tt3b.SetLineColor(ROOT.kBlack)
            tt3b.SetFillColor(ROOT.kAzure-9)
            tt3b.Draw("HIST SAME")
        data3b.SetStats(0)
        data3b.SetMarkerStyle(20)
        data3b.SetMarkerSize(0.7)
        data3b.Draw("P EX0 SAME axis")
        data3b.Draw("P EX0 SAME")
        plotName = args.outputDir+"/"+var+"_"+cut+"_3b.pdf" 
        print( plotName)
        c.SetLogy(True)
        c.SaveAs(plotName)

    return (data4b, tt4b, qcd4b, data3b, tt3b, qcd3b)





def loadROOTHists():

    data3bHist = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//dataRunII/hists_3b_newSBDef.root"
    data4bHist = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//dataRunII/hists_4b_newSBDef.root"
    tt3bHists  = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//TTRunII/hists_3b_newSBDef.root" 
    tt4bHists  = "root://cmseos.fnal.gov//store/user/jda102/condor/ZH4b/ULTrig//TTRunII/hists_4b_newSBDef.root"
    
    inFile3b = ROOT.TFile.Open(data3bHist)
    inFile4b = ROOT.TFile.Open(data4bHist)
    ttFile3b = ROOT.TFile.Open(tt3bHists)
    ttFile4b = ROOT.TFile.Open(tt4bHists)
    
    jetCombinatoricModelRoot.cd()

    (data4b, tt4b, qcd4b, data3b, tt3b, qcd3b)   = getHists(cut,args.weightRegion,"nSelJetsUnweighted", inFile4b, inFile3b, ttFile4b, ttFile3b)
    (data4b_nTagJets, tt4b_nTagJets, _, _, _, _) = getHists(cut,args.weightRegion,"nPSTJetsUnweighted", inFile4b, inFile3b, ttFile4b, ttFile3b)
    (_, _, _, _, _, qcd3b_nTightTags)            = getHists(cut,args.weightRegion,"nTagJetsUnweighted", inFile4b, inFile3b, ttFile4b, ttFile3b)
    print(f"data4b is {data4b}" )
    return data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags


do_read_coffea = True

if do_read_coffea:
    data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags = loadCoffeaHists()
else:
    data4b, data3b, tt4b, tt3b, qcd4b, qcd3b, data4b_nTagJets, tt4b_nTagJets, qcd3b_nTightTags = loadROOTHists()



#can = ROOT.TCanvas("test", 'test')
#qcd3b_nTightTags.Draw("histe")
#data3b_nTagJets.Draw("histe same")
#tt3b_nTagJets.Draw("histe same")
#can.SaveAs("tmp/test.png")
#sys.exit(0)


print("nSelJetsUnweighted", "data4b.Integral()", data4b.Integral(), "data3b.Integral()", data3b.Integral())
if tt4b and tt3b:
    print("nSelJetsUnweighted", "  tt4b.Integral()",   tt4b.Integral(),   "tt3b.Integral()",   tt3b.Integral())

print('data4b.Integral()',data4b.Integral())
print('data3b.Integral()',data3b.Integral())
if tt4b:
    print('  tt4b.Integral()',  tt4b.Integral())
if tt3b:
    print('  tt3b.Integral()',  tt3b.Integral())

mu_qcd = qcd4b.Integral()/qcd3b.Integral()
n4b = data4b.Integral()

n5b_true = data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(5))
data4b.SetBinContent(data4b.GetXaxis().FindBin(0), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(4)))
data4b.SetBinContent(data4b.GetXaxis().FindBin(1), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(5)))
data4b.SetBinContent(data4b.GetXaxis().FindBin(2), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(6)))
data4b.SetBinContent(data4b.GetXaxis().FindBin(3), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(7)))

data4b.SetBinError(data4b.GetXaxis().FindBin(0), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(4))**0.5)
data4b.SetBinError(data4b.GetXaxis().FindBin(1), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(5))**0.5)
data4b.SetBinError(data4b.GetXaxis().FindBin(2), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(6))**0.5)
data4b.SetBinError(data4b.GetXaxis().FindBin(3), data4b_nTagJets.GetBinContent(data4b_nTagJets.GetXaxis().FindBin(7))**0.5)

if tt4b:
    tt4b.SetBinContent(tt4b.GetXaxis().FindBin(0), tt4b_nTagJets.GetBinContent(tt4b_nTagJets.GetXaxis().FindBin(4)))
    tt4b.SetBinContent(tt4b.GetXaxis().FindBin(1), tt4b_nTagJets.GetBinContent(tt4b_nTagJets.GetXaxis().FindBin(5)))
    tt4b.SetBinContent(tt4b.GetXaxis().FindBin(2), tt4b_nTagJets.GetBinContent(tt4b_nTagJets.GetXaxis().FindBin(6)))
    tt4b.SetBinContent(tt4b.GetXaxis().FindBin(3), tt4b_nTagJets.GetBinContent(tt4b_nTagJets.GetXaxis().FindBin(7)))

    tt4b.SetBinError(tt4b.GetXaxis().FindBin(0), tt4b_nTagJets.GetBinError(tt4b_nTagJets.GetXaxis().FindBin(4)))
    tt4b.SetBinError(tt4b.GetXaxis().FindBin(1), tt4b_nTagJets.GetBinError(tt4b_nTagJets.GetXaxis().FindBin(5)))
    tt4b.SetBinError(tt4b.GetXaxis().FindBin(2), tt4b_nTagJets.GetBinError(tt4b_nTagJets.GetXaxis().FindBin(6)))
    tt4b.SetBinError(tt4b.GetXaxis().FindBin(3), tt4b_nTagJets.GetBinError(tt4b_nTagJets.GetXaxis().FindBin(7)))

threeTightTagFraction = qcd3b_nTightTags.GetBinContent(qcd3b_nTightTags.FindBin(3)) / qcd3b_nTightTags.Integral()
print("threeTightTagFraction",threeTightTagFraction)

def nTagPred(par,n):
    if tt4b_nTagJets:
        b = tt4b_nTagJets.GetXaxis().FindBin(n)
        nPred = tt4b_nTagJets.GetBinContent(b)
        nPredError = tt4b_nTagJets.GetBinError(b)**2
    else:
        b = 0
        nPred = 0
        nPredError = 0

    # nPred = 0
    # for bin in range(1,qcd3b.GetSize()-1):
    #     nj = int(qcd3b.GetBinCenter(bin))
    #     if nj < n: continue
    for nj in range(n,14):
        bin = qcd3b.GetXaxis().FindBin(nj)
        #w, nPseudoTagProb = getCombinatoricWeight(nj, par[0],par[1],par[2])#,par[3],par[4],par[5],par[6])
        w, nPseudoTagProb = getCombinatoricWeight(nj, par[0],par[1],par[2],threeTightTagFraction)
        nPred += nPseudoTagProb[n-3] * qcd3b.GetBinContent(bin)
        nPredError += (nPseudoTagProb[n-3] * qcd3b.GetBinError(bin))**2
        #nPred += nPseudoTagProb[n-3] * (data3b.GetBinContent(bin) - tt3b.GetBinContent(bin))
    nPredError = nPredError**0.5
    return nPred, nPredError

def bkgd_func_njet(x,par):
    nj = int(x[0] + 0.5)
    if nj in [0,1,2,3]:
        nTags = nj+4
        nEvents, _ = nTagPred(par,nTags)
        return nEvents

    if nj < 4: return 0
    w, _ = getCombinatoricWeight(nj, par[0],par[1],par[2],threeTightTagFraction)
    b = qcd3b.GetXaxis().FindBin(x[0])
    if tt4b:
        return w*qcd3b.GetBinContent(b) + tt4b.GetBinContent(b)
    return w*qcd3b.GetBinContent(b)


jetCombinatoricModels[cut] = jetCombinatoricModel()
jetCombinatoricModels[cut].threeTightTagFraction.fix = threeTightTagFraction

# set to prefit scale factor
#tf1_bkgd_njet = ROOT.TF1("tf1_bkgd",bkgd_func_njet,-0.5,14.5, jetCombinatoricModels[cut].nParameters)
tf1_bkgd_njet = ROOT.TF1("tf1_bkgd",bkgd_func_njet,0.,15., jetCombinatoricModels[cut].nParameters) # including the nbtags==4 bin in the fit double counts the normalization stat error
#tf1_bkgd_njet = ROOT.TF1("tf1_bkgd",bkgd_func_njet,0.5,14.5, jetCombinatoricModels[cut].nParameters) # including the nbtags==4 bin in the fit double counts the normalization stat error
#tf1_bkgd_njet = ROOT.TF1("tf1_qcd",bkgd_func_njet,3.5,11.5,3)

for parameter in jetCombinatoricModels[cut].parameters:
    tf1_bkgd_njet.SetParName(parameter.index, parameter.name)
    tf1_bkgd_njet.SetParLimits(parameter.index, parameter.lowerLimit, parameter.upperLimit)
    tf1_bkgd_njet.SetParameter(parameter.index, parameter.default)
    if parameter.fix is not None:
        tf1_bkgd_njet.FixParameter(parameter.index, parameter.fix)

## So that fit includes stat error from background templates, combine all stat error in quadrature
for ibin in range(1, data4b.GetSize()-2):#data4b.GetNbinsX()-2): #GetXaxis().GetNbins()-2):
    x = data4b.GetBinCenter(ibin)
    data4b_error = data4b.GetBinError(ibin)
    mu_qcd_this_bin = qcd4b.GetBinContent(ibin)/qcd3b.GetBinContent(ibin) if qcd3b.GetBinContent(ibin) else 0
    data3b_error = data3b.GetBinError(ibin) * mu_qcd_this_bin

    if tt4b:
        tt4b_error = tt4b.GetBinError(ibin)
    else:
        tt4b_error = 0

    if tt3b:
        tt3b_error = tt3b.GetBinError(ibin)
    else:
        tt3b_error = 0

    if tt4b and tt3b:
        total_error = (data3b_error**2 + data4b_error**2 + tt3b_error**2 + tt4b_error**2)**0.5 if data4b_error else 0
    elif tt4b:
        total_error = (data3b_error**2 + data4b_error**2  + tt4b_error**2)**0.5 if data4b_error else 0
    elif tt3b:
        total_error = (data3b_error**2 + data4b_error**2  + tt3b_error**2)**0.5 if data4b_error else 0
    else:
        total_error = (data3b_error**2 + data4b_error**2  )**0.5 if data4b_error else 0

    increase = 100*total_error/data4b_error if data4b_error else 100
    if tt4b and tt3b:
        print('%2i, %2.0f| %5.1f, %5.1f, %5.1f, %5.1f, %5.0f%%'%(ibin, x, data4b_error, data3b_error, tt4b_error, tt3b_error, increase))
    elif tt4b:
        print('%2i, %2.0f| %5.1f, %5.1f, %5.1f, %5.0f%%'%(ibin, x, data4b_error, data3b_error, tt4b_error, increase))
    elif tt3b:
        print('%2i, %2.0f| %5.1f, %5.1f, %5.1f, %5.0f%%'%(ibin, x, data4b_error, data3b_error, tt3b_error, increase))
    else:
        print('%2i, %2.0f| %5.1f, %5.1f, %5.0f%%'%(ibin, x, data4b_error, data3b_error, increase))

    data4b.SetBinError(ibin, total_error)

# perform fit
data4b.Fit(tf1_bkgd_njet,"0R L")
chi2 = tf1_bkgd_njet.GetChisquare()
ndf = tf1_bkgd_njet.GetNDF()
prob = tf1_bkgd_njet.GetProb()
print("chi^2 =",chi2,"ndf =",ndf,"chi^2/ndf =",chi2/ndf,"| p-value =",prob)

print("Pulls:")
for bin in range(1,data4b.GetSize()-2):
    error = data4b.GetBinError(bin)
    residual = data4b.GetBinContent(bin)-tf1_bkgd_njet.Eval(data4b.GetBinCenter(bin))
    pull = residual/error if error else 0
    print('%2i| %5.1f/%5.1f = %4.1f'%(bin, residual, error, pull))

for parameter in jetCombinatoricModels[cut].parameters:
    parameter.value = tf1_bkgd_njet.GetParameter(parameter.index)
    parameter.error = tf1_bkgd_njet.GetParError( parameter.index)

jetCombinatoricModels[cut].dump()
for parameter in jetCombinatoricModels[cut].parameters:
    jetCombinatoricModelFile.write(parameter.name+"_"+cut+"               "+str(parameter.value)+"\n")
    jetCombinatoricModelFile.write(parameter.name+"_"+cut+"_err           "+str(parameter.error)+"\n")
    jetCombinatoricModelFile.write(parameter.name+"_"+cut+"_pererr        "+str(parameter.percentError)+"\n")
jetCombinatoricModelFile.write("chi^2     "+str(chi2)+"\n")
jetCombinatoricModelFile.write("ndf       "+str(ndf)+"\n")
jetCombinatoricModelFile.write("chi^2/ndf "+str(chi2/ndf)+"\n")
jetCombinatoricModelFile.write("p-value   "+str(prob)+"\n")

n5b_pred, n5b_pred_error = nTagPred(tf1_bkgd_njet.GetParameters(),5)
print("Fitted number of 5b events: %5.1f +/- %f"%(n5b_pred, n5b_pred_error))
print("Actual number of 5b events: %5.1f, (%3.1f sigma pull)"%(n5b_true,(n5b_true-n5b_pred)/n5b_pred**0.5))
jetCombinatoricModelFile.write("n5b_pred   "+str(n5b_pred)+"\n")
jetCombinatoricModelFile.write("n5b_true   "+str(n5b_true)+"\n")


background_TH1 = data4b.Clone("background_TH1")
background_TH1.Reset()

# Reset bin error for plotting
for bin in range(1,data4b.GetSize()-2):
    if data4b.GetBinContent(bin) > 0:
        data4b_error = data4b.GetBinContent(bin)**0.5
        data4b.SetBinError(bin, data4b_error)

    binCenter = int(background_TH1.GetBinCenter(bin))
    bc = tf1_bkgd_njet.Eval(binCenter)
    background_TH1.SetBinContent(bin, bc)
    if binCenter < 4:
        bc, be = nTagPred(tf1_bkgd_njet.GetParameters(), binCenter+4)
    else:
        te = tt4b.GetBinError(bin) if tt4b else 0
        qc = qcd3b.GetBinContent(bin)
        qe = qcd3b.GetBinError(bin)
        be = (te**2 + (qe*bc/qc if qc else 0)**2)**0.5
    background_TH1.SetBinError(bin, be)
background_TH1.Write()

c=ROOT.TCanvas(cut+"_postfit_tf1","Post-fit")
#data4b.SetLineColor(ROOT.kBlack)
data4b.GetYaxis().SetTitleOffset(1.5)
data4b.GetYaxis().SetTitle("Events")
xTitle = "Number of b-tags - 4"+" "*63+"Number of Selected Jets"
data4b.GetXaxis().SetTitle(xTitle)
data4b.Draw("P EX0")
data4b.Write()
qcdDraw = ROOT.TH1F(qcd3b)
qcdDraw.SetName(qcd3b.GetName()+"draw")
qcd3b.Write()

stack = ROOT.THStack("stack","stack")
#mu_qcd = qcd4b.Integral()/qcdDraw.Integral()
print("mu_qcd = %f +/- %f%%"%(mu_qcd, 100*n4b**-0.5))
jetCombinatoricModelFile.write("mu_qcd_"+cut+"       "+str(mu_qcd)+"\n")
qcdDraw.Scale(mu_qcd)
qcdDraw.SetLineColor(ROOT.kMagenta)
#stack.Add(qcdDraw,"hist")
#stack.Draw("HIST SAME")
if tt4b:
    stack.Add(tt4b)
    tt4b.Write()
stack.Add(qcdDraw)
#qcdDraw.Write()
stack.Draw("HIST SAME")
#qcd3b.Draw("HIST SAME")
data4b.SetStats(0)
data4b.SetMarkerStyle(20)
data4b.SetMarkerSize(0.7)
data4b.Draw("P EX0 SAME axis")
data4b.Draw("P EX0 SAME")
background_TH1.SetLineColor(ROOT.kRed)
background_TH1.Draw("HIST SAME")
#tf1_bkgd_njet.SetLineColor(ROOT.kRed)
#tf1_bkgd_njet.Draw("SAME")
tf1_bkgd_njet.Write()

xleg, yleg = [0.67, 0.9-0.035], [0.9-0.06*4, 0.9-0.035]
leg = ROOT.TLegend(xleg[0], yleg[0], xleg[1], yleg[1])
leg.AddEntry(data4b, "Data "+str(lumi)+"/fb, "+args.year)
leg.AddEntry(qcdDraw, "Multijet Model")
if tt4b:
    leg.AddEntry(tt4b, "t#bar{t}")
leg.AddEntry(background_TH1, "JCM Fit")
#leg.AddEntry(tf1_bkgd_njet, "JCM Fit")
leg.Draw()

c.Update()
print(c.GetFrame().GetY1(),c.GetFrame().GetY2())
line=ROOT.TLine(3.5,-5000,3.5,c.GetFrame().GetY2())
line.SetLineColor(ROOT.kBlack)
line.Draw()
histName = args.outputDir+"/"+"nJets_"+cut+"_postfit_tf1.pdf"
print(histName)
c.SaveAs(histName)


jetCombinatoricModelFile.close()

# To add

##
##    #jetCombinatoricModelRoot.Close()
##
##    samples=collections.OrderedDict()
##    samples[JCMROOTFileName] = collections.OrderedDict()
##    samples[JCMROOTFileName][data4b.GetName()] = {
##        "label" : ("Data %.1f/fb, "+o.year)%(lumi),
##        "legend": 1,
##        "isData" : True,
##        "ratio" : "numer A",
##        "color" : "ROOT.kBlack"}
##    samples[JCMROOTFileName][qcd3b.GetName()] = {
##        "label" : "Multijet Model",
##        "weight": mu_qcd,
##        "legend": 2,
##        "stack" : 3,
##        "ratio" : "denom A",
##        "color" : "ROOT.kYellow"}
##    if tt4b:
##        samples[JCMROOTFileName][tt4b.GetName()] = {
##            "label" : "t#bar{t}",
##            "legend": 3,
##            "stack" : 2,
##            "ratio" : "denom A",
##            "color" : "ROOT.kAzure-9"}
##    #samples[JCMROOTFileName][tf1_bkgd_njet.GetName()] = {
##    samples[JCMROOTFileName]["background_TH1"] = {
##        "label" : "JCM Fit",
##        "legend": 4,
##        "ratio": "denom A", 
##        "color" : "ROOT.kRed"}
##
##    #xTitle = "Number of b-tags - 4"+" "*31+"Number of Selected Jets"
##    xTitle = "Extra b-tags"+" "*36+"Number of Selected Jets"
##    parameters = {"titleLeft"   : "#bf{CMS} Internal",
##                  "titleCenter" : regionNames[o.weightRegion],
##                  "titleRight"  : cutTitle,
##                  "maxDigits"   : 4,
##                  "ratio"     : True,
##                  "rMin"      : 0,
##                  "rMax"      : 2,
##                  "xMin"      : 0.5,
##                  "xMax"      : 14.5,
##                  "rTitle"    : "Data / Bkgd.",
##                  "xTitle"    : xTitle,
##                  "yTitle"    : "Events",
##                  "legendSubText" : ["",
##                                     "#bf{Fit Result:}",
##                                     "#font[12]{f} = %0.3f #pm %0.1f%%"%(jetCombinatoricModels[cut].pseudoTagProb.value, jetCombinatoricModels[cut].pseudoTagProb.percentError),
##                                     "#font[12]{e} = %0.2f #pm %0.1f%%"%(jetCombinatoricModels[cut].pairEnhancement.value, jetCombinatoricModels[cut].pairEnhancement.percentError),
##                                     "#font[12]{d} = %0.2f #pm %0.1f%%"%(jetCombinatoricModels[cut].pairEnhancementDecay.value, jetCombinatoricModels[cut].pairEnhancementDecay.percentError),
##                                     "#chi^{2}/DoF = %0.2f"%(chi2/ndf),
##                                     "p-value = %2.0f%%"%(prob*100),
##                                     ],
##                  "outputDir" : o.outputDir,
##                  "outputName": "nSelJets"+st+"_"+cut+"_postfit_tf1"}
##
##    PlotTools.plot(samples, parameters)
