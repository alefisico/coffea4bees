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


def plot(var, **kwargs):
    makePlot(cfg, var, outputFolder= args.outputFolder, **kwargs)
    plt.close()

def doPlots(varList, debug=False):


    # Fig 42
    plot("FvT_noFvT",region="SB",yscale="linear",norm=0,rebin=2,doratio=1,rlim=[0.5,1.5])
    plot("FvT.FvT",  region="SB",yscale="linear",norm=0,rebin=2,doratio=1,rlim=[0.5,1.5])





if __name__ == '__main__':

    args = parse_args()

    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder

    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    if args.list_of_hists:
        varList = args.list_of_hists
    else:
        varList = [h for h in cfg.hists[0].keys() if not any(skip in h for skip in args.skip_hists)]
    doPlots(varList, debug=args.debug)
