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
    #splittings = ["bb", "bj", "b(bj)", ]

    splittings = [i.replace("splitting_","").replace(".pt_l","").replace("_","/") for i in cfg.hists[0]["hists"].keys() if not i.find("pt_l") == -1 and i.find("detailed") == -1]

    for _year in ["RunII","UL18","UL17","UL16_preVFP","UL16_postVFP"]:

        for _split in splittings:


            #
            #  config Setup
            #

            args = {"norm": True,
                    "doRatio": 1,
                    "region":"sum",
                    "cut":"passPreSel",
                    "rebin":1,
                    "year":_year,
                    #"process":"data",
                    #"histtype":"step",
                    }

            _split_name = "splitting_" + _split


            plot(f"{_split_name}.mA",          **args)
            plot(f"{_split_name}.mA_l",        **args)
            plot(f"{_split_name}.mA_vl",        **args)
            plot(f"{_split_name}.mB",          **args)
            plot(f"{_split_name}.mB_l",        **args)
            plot(f"{_split_name}.mB_vl",        **args)

            plot(f"{_split_name}.pt_l",        **args)
            plot(f"{_split_name}.n",           **args)

            args["rebin"] = 2
            plot(f"{_split_name}.decay_phi",   **args)
            plot(f"{_split_name}.zA",          **args)
            plot(f"{_split_name}.zA_l",          **args)
            plot(f"{_split_name}.thetaA",      **args)

            #plot(f"{_split_name}.eta",         **args)

            # args["doRatio"] = 0
            # plot(f"{_split_name_0}.n", **args)
            # plot(f"{_split_name_1}.n", **args)


            plot2d(var=f"{_split_name}.zA_l_vs_thetaA_pT", region="sum", cut="passPreSel",doRatio=0,rebin=1,process="data"  , year=_year)
            plot2d(var=f"{_split_name}.zA_l_vs_thetaA_pT", region="sum", cut="passPreSel",doRatio=0,rebin=1,process="syn_v0", year=_year)

        #
        # add vs PT plots ?
        #


if __name__ == '__main__':

    args = parse_args()
    print(args.metadata)
    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files
    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    #varList = [ h for h in cfg.hists[0]['hists'].keys() if not h in args.skip_hists ]
    doPlots(debug=args.debug)
