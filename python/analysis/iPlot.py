import os, sys
import hist
import argparse
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coffea.util import load
from hist.intervals import ratio_uncertainty

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



#
# TO Add
#     Variable Binning
#  - labels


def ls(option="var", match=None):
    for k in axisLabels[option]:
        if match:
            if k.find(match) != -1: print(k)
        else:
            print(k)
        

def plot(var='selJets.pt',year="2017",cut="passPreSel",tag="fourTag",region="SR", **kwargs):
    r"""
    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       year     : "2017",
       cut      : "passPreSel",
       tag      : "fourTag",
       region   : "SR",
    
       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    if kwargs.get("debug",False): print(f'kwargs = {kwargs}')
    
    if var.find("*") != -1:
        ls(match=var.replace("*",""))
        return 

    fig = makePlot(hists, cutList, codeDicts, var=var, year=year, cut=cut, tag=tag, region=region, outputFolder=args.outputFolder, **kwargs)


    fileName = "test.pdf"
    fig.savefig(fileName)
    plt.close()
    os.system("open "+fileName)
    
        
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

                

