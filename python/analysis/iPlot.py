import os
import sys
import yaml
import hist
import argparse
import matplotlib.pyplot as plt
from coffea.util import load
from hist.intervals import ratio_uncertainty
sys.path.insert(0, os.getcwd())
from base_class.plots import makePlot, make2DPlot
import analysis.iPlot_config as cfg

#
# TO Add
#     Variable Binning


def ls(option="var", match=None):
    for k in cfg.axisLabels[option]:
        if match:
            if k.find(match) != -1:
                print(k)
        else:
            print(k)

def examples():
    print("examples:")
    #>>> plot("*",doRatio=1, region="SR",cut="passPreSel",rlim=[0,2],norm=0, process="Multijet")
    #
    #>>> plot("selElecs.pt",doRatio=1, region="SR",cut="passPreSel",rlim=[0,2],norm=0, process="Multijet")
    #
    #>>> plot("v4j.mass",doRatio=1, region="SR",cut=["passPreSel","failSvB","passSvB"],rebin=4,rlim=[0,2],norm=0, process="Multijet",debug=True,yscale="log")
    #>>> plot2d("quadJet_min_dr.close_vs_other_m",process="HH4b",region="SR",cut="passPreSel")

            
def plot(var='selJets.pt', *, cut="passPreSel", region="SR", **kwargs):
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

    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    if var.find("*") != -1:
        ls(match=var.replace("*", ""))
        return

    if len(cfg.hists) > 1:
        fig = makePlot(cfg.hists, cfg.cutList, cfg.plotConfig, var=var, cut=cut, region=region,
                       outputFolder=cfg.outputFolder, fileLabels=args.fileLabels, **kwargs)
    else:
        fig = makePlot(cfg.hists[0], cfg.cutList, cfg.plotConfig, var=var, cut=cut, region=region,
                       outputFolder=cfg.outputFolder, **kwargs)
    #breakpoint()
    fileName = "test.pdf"
    fig.savefig(fileName)
    plt.close()
    os.system("open "+fileName)


def plot2d(var='quadJet_selected.lead_vs_subl_m', process="HH4b",
           *, cut="passPreSel", region="SR", **kwargs):
    r"""
    Plot 2d

    Call with:
       plot2d("quadJet_selected.lead_vs_subl_m",process="data",region="SB",cut="passPreSel",tag="threeTag")
       plot2d("quadJet_selected.lead_vs_subl_m",process="HH4b",region="SR",cut="passPreSel",tag="threeTag")


    Takes Options:

       debug    : False,
       var      : 'quadJet_selected.lead_vs_subl_m',
       process  : 'HH4b',
       cut      : "passPreSel",
       region   : "SR",

       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    if var.find("*") != -1:
        ls(match=var.replace("*", ""))
        return

    fig = make2DPlot(cfg.hists[0], process, cfg.cutList, cfg.plotConfig, var=var, cut=cut,
                     region=region, outputFolder=cfg.outputFolder, **kwargs)

    fileName = "test.pdf"
    fig.savefig(fileName)
    plt.close()
    os.system("open "+fileName)


def parse_args():
    
    parser = argparse.ArgumentParser(description='uproot_plots')

    parser.add_argument('-i', '--inputFile', dest="inputFile",
                        default='hists.pkl', nargs='+',
                        help='Input File. Default: hists.pkl')

    parser.add_argument('-l', '--labelNames', dest="fileLabels",
                        default=["fileA", "fileB"], nargs='+',
                        help='label Names when more than one input file')

    
    parser.add_argument('-o', '--outputFolder', default=None,
                        help='Folder for output folder. Default: plots/')

    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="analysis/metadata/plotsNominal.yml",
                        help='Metadata file.')

    args = parser.parse_args()
    return args


def load_config(metadata):
    """  Load meta data
    """
    plotConfig = yaml.safe_load(open(metadata, 'r'))

    #
    #  Make two way code mapping:
    #    ie: 3 mapts to  "threeTag" and "threeTag" maps to 3
    for k, v in plotConfig["codes"]["tag"].copy().items():
        plotConfig["codes"]["tag"][v] = k

    for k, v in plotConfig["codes"]["region"].copy().items():
        plotConfig["codes"]["region"][v] = k

    return plotConfig


def load_hists(input_hists):
    hists = []
    for _inFile in input_hists:
        with open(_inFile, 'rb') as hfile:
            hists.append(load(hfile))

    return hists


def read_axes_and_cuts(hists, plotConfig):

    axisLabels = {}
    cutList = []
    
    axisLabels["var"] = hists[0]['hists'].keys()
    var1 = list(hists[0]['hists'].keys())[0]

    for a in hists[0]["hists"][var1].axes:
        axisName = a.name
        if axisName == var1:
            continue

        if isinstance(a, hist.axis.Boolean):
            print(f"Adding cut\t{axisName}")
            cutList.append(axisName)
            continue

        if a.extent > 20:
            continue   # HACK to skip the variable bins FIX

        axisLabels[axisName] = []
        print(axisName)
        for iBin in range(a.extent):
            if axisName in plotConfig["codes"]:
                value = plotConfig["codes"][axisName][a.value(iBin)]

            else:
                value = a.value(iBin)

            print(f"\t{value}")
            axisLabels[axisName].append(value)

    return axisLabels, cutList


if __name__ == '__main__':

    args = parse_args()
    
    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder
    
    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)

    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    
