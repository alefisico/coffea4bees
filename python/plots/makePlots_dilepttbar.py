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

    for v in varList:

        if debug: print(f"plotting 1D ...{v}")

        vDict = cfg.plotModifiers.get(v, {})
        if vDict.get("2d", False):
            continue

        cut = "passDilepTtbar"
        tag = None

        vDict["ylabel"] = "Entries"
        vDict["doRatio"] = True
        vDict["legend"] = True

        if args.doTest:
            vDict["write_yaml"] = True


        if debug: print(f"plotting 1D ...{v}")
        plot_args  = {}
        plot_args["var"] = v
        plot_args["cut"] = cut
        plot_args["outputFolder"] = args.outputFolder
        plot_args['debug'] = debug
        plot_args['region'] = 'dileptonic TTbar Selection'
        plot_args = plot_args | vDict
        if debug: print(plot_args)
        try:
            fig = makePlot(cfg, **plot_args)
        except ValueError:
            print(f"ValueError: {v} {region}")
            pass

        plt.close()
    
if __name__ == '__main__':

    args = parse_args()

    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder

    # cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))
    cfg.plotModifiers = { 'tagJets_dilepttbar.n' : {"xlim":[0,4], 'xlabel': 'Number of btagged jets'}}

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.cutList = ['passDilepTtbar']
    cfg.hists[0]['hists'] = cfg.hists[0]['hists_ttbar'] 
    cfg.hists[0]['categories'] = ['process', 'year'] + cfg.cutList

    if args.list_of_hists:
        varList = args.list_of_hists
    else:
        varList = [h for h in cfg.hists[0]['hists_ttbar'].keys() if not any(skip in h for skip in args.skip_hists)]
    doPlots(varList, debug=args.debug)
