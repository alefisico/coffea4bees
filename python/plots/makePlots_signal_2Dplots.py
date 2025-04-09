import os
import time
import sys
import yaml
import hist
import argparse
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np

sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')

def doPlots(varList, debug=False):

    #
    #  2D Plots
    #
    for v in varList:
        print(v)

        vDict = {}
        vDict["ylabel"] = "Entries"
        vDict["doRatio"] = False #cfg.plotConfig.get("doRatio", True)
        vDict["legend"] = True

        if args.doTest:
            vDict["write_yaml"] = True

        # for process in ["data", "Multijet", "HH4b", "TTbar"]:
        #     for region in ["SR", "SB"]:

        plot_args  = {}
        plot_args["var"] = v
        plot_args["outputFolder"] = args.outputFolder
        if 'leadstmass_vs_sublstmass' in v: plot_args["plot_contour"] = True
        if 'leadstdr_vs_m4j' in v: plot_args["plot_leadst_lines"] = True
        if 'sublstdr_vs_m4j' in v: plot_args["plot_sublst_lines"] = True
        plot_args = plot_args | vDict

        if debug: print(plot_args)

        fig = make2DPlot(cfg, 'process',
                            **plot_args)
        plt.close()

    
if __name__ == '__main__':

    args = parse_args()

    cfg.plotConfig = load_config(args.metadata)
    cfg.plotConfig['hist_dict'] = { 'process': sum, 'selection': 'none', 'year': 'UL18' }
    # cfg.plotConfig['hist_dict'] = { 'process': 'GluGluToHHTo4B_cHHH0', 'selection': 'none', 'year': 'UL18' }
    cfg.outputFolder = args.outputFolder

    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    # cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    if args.list_of_hists:
        varList = args.list_of_hists
    else:
        varList = [h for h in cfg.hists[0].keys() if not any(skip in h for skip in args.skip_hists)]

    cutList = ['none', 'none_SBSR', 'none_SR', 'passDiJetMass', 'passDiJetMass_SBSR', 'passDiJetMass_SR', 'passDiJetMassOneMDR', 'passDiJetMassOneMDR_SBSR', 'passDiJetMassOneMDR_SR', 'passDiJetMassMDR', 'passDiJetMassMDR_SBSR', 'passDiJetMassMDR_SR', 'selected', 'selected_SBSR', 'selected_SR']
    # cutList = ['selected_SR']

    for isel in cutList:
        cfg.plotConfig['hist_dict']['selection'] = isel
        doPlots(varList, debug=args.debug)
