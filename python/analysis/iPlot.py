import os, sys
import yaml
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
# TO Add
#     Variable Binning
#  - labels


def ls(option="var", match=None):
    for k in axisLabels[option]:
        if match:
            if k.find(match) != -1: print(k)
        else:
            print(k)
        

def plot(var='selJets.pt', *, cut="passPreSel",region="SR", **kwargs):
    r"""
    Plot


    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       cut      : "passPreSel",
       region   : "SR",
    
       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    if kwargs.get("debug",False): print(f'kwargs = {kwargs}')
    
    if var.find("*") != -1:
        ls(match=var.replace("*",""))
        return 

    fig = makePlot(hists, cutList, plotConfig, var=var, cut=cut, region=region, outputFolder=args.outputFolder, **kwargs)


    fileName = "test.pdf"
    fig.savefig(fileName)
    plt.close()
    os.system("open "+fileName)
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', dest="inputFile", default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o','--outputFolder', default=None, help='Folder for output folder. Default: plots/')
    parser.add_argument('-m','--metadata', dest="metadata", default="analysis/metadata/plotsNominal.yml", help='Metadata file.')
    args = parser.parse_args()

    plotConfig = yaml.safe_load(open(args.metadata, 'r'))  
    for k, v in plotConfig["codes"]["tag"].copy().items():
        plotConfig["codes"]["tag"][v] = k
    for k, v in plotConfig["codes"]["region"].copy().items():
        plotConfig["codes"]["region"][v] = k
    
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
                if axisName in plotConfig["codes"]:
                    value = plotConfig["codes"][axisName][a.value(iBin)]

                else:
                    value = a.value(iBin)

                print(f"\t{value}")
                axisLabels[axisName].append(value)

                

