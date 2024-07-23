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
import yaml

sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')

def plot(var, **kwargs):
    fig, ax = makePlot(cfg, var, outputFolder= args.outputFolder, **kwargs)
    plt.close()
    return fig, ax


def plot2d(process, **kwargs):
    fig, ax = make2DPlot(cfg, process, outputFolder= args.outputFolder, **kwargs)
    plt.close()
    return fig, ax


labels = {"bb":"g $\\rightarrow$ bb",
          "bj":"b $\\rightarrow$ bj",
          "b(bj)":"g $\\rightarrow$ b (bj)",
          }




def doPlots(debug=False):



    #
    #  Compare Splittings
    #
    splittings = ["bb", "bj", "b(bj)", ]

    for _split in splittings:

    
        #
        #  config Setup
        #

        args = {"norm": True,
                "doRatio": 1,
                "labels":["clustered", "re-clustered"],
                "norm": True,
                "region":"SR",
                "cut":"passPreSel",
                "doRatio":0,
                "rebin":1,
                "process":"data",
                "histtype":"step",
                }

        _split_name_0 = "splitting_" + _split
        _split_name_1 = "splitting_" + _split + "_re"


        plot([f"{_split_name_0}.drAB",       f"{_split_name_1}.drAB"],      **args)
        plot([f"{_split_name_0}.thetaA",     f"{_split_name_1}.thetaA"],    **args)
        plot([f"{_split_name_0}.decay_phi",  f"{_split_name_1}.decay_phi"], **args)
        plot([f"{_split_name_0}.mA",         f"{_split_name_1}.mA"],        **args)
        plot([f"{_split_name_0}.mB",         f"{_split_name_1}.mB"],        **args)
        plot([f"{_split_name_0}.zA",         f"{_split_name_1}.zA"],        **args)
        plot([f"{_split_name_0}.pt_l",       f"{_split_name_1}.pt_l"],      **args)
        plot([f"{_split_name_0}.eta",        f"{_split_name_1}.eta"],       **args)
    
        plot(f"{_split_name_0}.n", **args)
        plot(f"{_split_name_1}.n", **args)    
    
    
        plot2d(var=f"{_split_name_1}.zA_vs_thetaA", region="SR", cut="passPreSel",doRatio=0,rebin=1,process="data",histtype="step",debug=0,norm=1)
        plot2d(var=f"{_split_name_0}.zA_vs_thetaA", region="SR", cut="passPreSel",doRatio=0,rebin=1,process="data",histtype="step",debug=0,norm=1)

        #
        # add vs PT plots ? 
        #
        
    
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

    #varList = [ h for h in cfg.hists[0]['hists'].keys() if not h in args.skip_hists ]
    doPlots(debug=args.debug)
