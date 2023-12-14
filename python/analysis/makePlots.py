import os
import time
import sys
import yaml
import hist
import argparse
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np

sys.path.insert(0, os.getcwd())
from base_class.plots import makePlot


def doPlots(varList, cutList):

    if args.doTest:
        varList = ["SvB_MA.ps_zz", "SvB_MA.ps_zh", "SvB_MA.ps_hh"]

    for v in varList:
        print(v)

        vDict = plotModifiers.get(v, {})

        cut = "passPreSel"
        tag = "fourTag"

        vDict["ylabel"] = "Entries"
        vDict["doRatio"] = plotConfig.get("rebin", "True")
        vDict["legend"] = True

        for region in ["SR", "SB"]:
            fig = makePlot(hists, cutList, plotConfig, var=v,
                           cut=cut, region=region,
                           outputFolder=args.outputFolder, **vDict)
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i', '--inputFile', dest="inputFile",
                        default='hists.pkl',
                        help='Input File. Default: hists.pkl')
    parser.add_argument('-o', '--outputFolder', default=None,
                        help='Folder for output folder. Default: plots/')
    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="analysis/metadata/plotsNominal.yml",
                        help='Metadata file.')
    parser.add_argument('--modifiers', dest="modifiers",
                        default="analysis/metadata/plotModifiers.yml",
                        help='Metadata file.')
    parser.add_argument('--doTest', action="store_true", help='Metadata file.')
    args = parser.parse_args()

    plotConfig = yaml.safe_load(open(args.metadata, 'r'))
    for k, v in plotConfig["codes"]["tag"].copy().items():
        plotConfig["codes"]["tag"][v] = k
    for k, v in plotConfig["codes"]["region"].copy().items():
        plotConfig["codes"]["region"][v] = k

    plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if args.outputFolder:
        if not os.path.exists(args.outputFolder):
            os.makedirs(args.outputFolder)

    with open(f'{args.inputFile}', 'rb') as hfile:
        hists = load(hfile)

        axisLabels = {}
        axisLabels["var"] = hists['hists'].keys()
        var1 = list(hists['hists'].keys())[0]

        varList = list(hists['hists'].keys())
        cutList = []

        for a in hists["hists"][var1].axes:
            axisName = a.name
            if axisName == var1:
                continue

            if isinstance(a, hist.axis.Boolean):
                print(f"Adding cut\t{axisName}")
                cutList.append(axisName)
                continue

            if a.extent > 20:
                continue    # HACK to skip the variable bins FIX
            axisLabels[axisName] = []
            print(axisName)
            for iBin in range(a.extent):
                if axisName in plotConfig["codes"]:
                    value = plotConfig["codes"][axisName][a.value(iBin)]

                else:
                    value = a.value(iBin)

                print(f"\t{value}")
                axisLabels[axisName].append(value)

        doPlots(varList, cutList)
