import os, time, sys
import hist
import argparse
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np

#sys.path.insert(0, '../') 
sys.path.insert(0,os.getcwd())
from base_class.plots import makePlot


#
# Move following to a config file?
#
axes = ["var","process","year","tag","region","cut"]
codeDicts = {}
codeDicts["tag"] = {"threeTag":3, "fourTag":4, 3:"threeTag", 4:"fourTag"}
codeDicts["region"]  = {"SR":2, "SB":1, 2:"SR", 1:"SB", 0:"other","other":0}


variables = {
    "SvB_MA_ps"      : {},
    "SvB_ps"         : {},

    "selJets.energy" : {},
    "selJets.eta" : {},
    "selJets.mass" : {"xlim":[0,100]},
    "selJets.n" : {},
    "selJets.phi" : {},
    "selJets.pt" : {'yscale':'log', 'xlim':[40,400],},
    "selJets.pz" : {},

    "canJets.energy" : {},
    "canJets.eta"  : {"xlim":[-3,3]},
    "canJets.mass" : {"xlim":[0,100]},
    "canJets.n" : {},
    "canJets.phi" : {},
    "canJets.pt" : {},
    "canJets.pz" : {},

    "othJets.energy" : {},
    "othJets.eta" : {},
    "othJets.mass" : {"xlim":[0,100]},
    "othJets.n" : {"xlim":[0,15]},
    "othJets.phi" : {},
    "othJets.pt" : {},
    "othJets.pz" : {},
}



def doPlots():
    for v, vDict in variables.items():
        print(v)

        
        year ="UL18"
        cut  = "passPreSel"
        tag  ="fourTag"


        vDict["ylabel"]  = "Entries"
        vDict["doRatio"] = True
        vDict["legend"]  = True

        #vDict["debug"] = True

        #vDict["norm"] = True
        #print(v,vDict)

        #,ylabel="Entries",,rebin=1,xlim=[40,400],rlim=[0.5,2])

        for region in ["SR","SB"]:
            fig = makePlot(hists, cutList, codeDicts, var=v, year=year, cut=cut, tag=tag, region=region, outputFolder=args.outputFolder, **vDict)
    
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', dest="inputFile", default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o','--outputFolder', default=None, help='Folder for output folder. Default: plots/')
    args = parser.parse_args()

    if args.outputFolder:
        if not os.path.exists(args.outputFolder): os.makedirs(args.outputFolder)
    

    with open(f'{args.inputFile}', 'rb') as hfile:
        hists = load(hfile)

        
        axisLabels = {}
        axisLabels["var"] = hists['hists'].keys()
        var1 = list(hists['hists'].keys())[0]

        cutList = []
        
        for a in hists["hists"][var1].axes:
            axisName = a.name
            if axisName == var1: continue

            if type(a) == hist.axis.Boolean:
                print(f"Adding cut\t{axisName}")
                cutList.append(axisName)
                continue

            if a.extent > 20: continue # HACK to skip the variable bins FIX
            axisLabels[axisName] = []
            print(axisName)
            for iBin in range(a.extent):
                if axisName in codeDicts:
                    value = codeDicts[axisName][a.value(iBin)]

                else:
                    value = a.value(iBin)
                    
                print(f"\t{value}")
                axisLabels[axisName].append(value)

                

        doPlots()
