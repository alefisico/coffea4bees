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
    print("examples:\n\n")
    print(
        '# Nominal plot of data and background in the a region passing a cut \n'
        'plot("v4j.mass", region="SR", cut="passPreSel")\n\n'

        '# Can get a print out of the varibales\n'
        'ls()'
        'plot("*", region="SR", cut="passPreSel")\n'
        'plot("v4j*", region="SR", cut="passPreSel")\n\n'

        '# Can add ratio\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1)\n\n'

        '# Can rebin\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4)\n\n'

        '# Can normalize\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1)\n\n'

        '# Can set logy\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, yscale="log")\n\n'

        '# Can set ranges\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, rlim=[0.5,1.5])\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, xlim=[0,1000])\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, ylim=[0,0.01])\n\n'

        '# Can overlay different regions \n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="data", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="HH4b", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="Multijet", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="TTToHadronic", doRatio=1, rebin=4)\n\n'

        '# Can overlay different cuts \n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="data", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="HH4b", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="Multijet", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="TTToHadronic", doRatio=1, rebin=4, norm=1)\n\n'

        '# Can plot a single process  \n'
        'plot("v4j.mass", region="SR", cut="passPreSel",process="data")

        '# Can overlay processes  \n'
        'plot("v4j.mass", region="SR", cut="passPreSel",norm=1,process=["data","TTTo2L2Nu","HH4b","Multijet"],doRatio=1)'

        '# Plot 2d hists \n'
        'plot2d("quadJet_min_dr.close_vs_other_m",process="Multijet",region="SR",cut="failSvB")\n'
        'plot2d("quadJet_min_dr.close_vs_other_m",process="Multijet",region="SR",cut="failSvB",full=True)\n\n'
    )

            
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
                       outputFolder=cfg.outputFolder, fileLabels=cfg.fileLabels, **kwargs)
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
                        default="analysis/metadata/plotsAll.yml",
                        help='Metadata file.')

    parser.add_argument('--modifiers', dest="modifiers",
                        default="analysis/metadata/plotModifiers.yml",
                        help='Metadata file.')

    parser.add_argument('--doTest', action="store_true", help='Metadata file.')
    
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
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    
