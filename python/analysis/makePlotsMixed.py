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

def doPlots(debug=False):

    #
    #  Try to get averages
    #
    var_to_plot "SvB_MA_FvT_3bDvTMix4bDvT_v1_newSBDef.ps_hh"
    _hist_config = {"process": "data_3b_for_mixed",
                    "year":  sum,
                    "tag":   hist.loc("threeTag"),
                    }
    region_dict = {"region":  hist.loc(codes["region"][region])}
    cut_dict    = get_cut_dict(cut, cfg.cutList)
    _hist_obj   = get_hist(input_data, var_to_plot, "data_3b_for_mixed")

    
    #
    #  SvB mixed v0 and v1 vs v2 bkg
    #
    var_list = ["SvB_MA.ps_hh", "SvB_MA.ps_zh", "SvB_MA.ps_zz", "SvB_MA.ps_hh_fine", "SvB_MA.ps_zh_fine", "SvB_MA.ps_zz_fine", ]
    rebin_list = [20, 10, 8, 20, 10, 8] 

    for var, rebin in zip(var_list, rebin_list):
        print(f"{var} (rebin {rebin})" )
        var_bkg = var.replace("SvB_MA","SvB_MA_FvT_3bDvTMix4bDvT_v1_newSBDef")
        
        plot_args = {"var":var,
                     "var_over_ride":{"Multijet": var_bkg},
                     }

        plot_args = plot_args | {"region":"SR", "cut":"passPreSel", "doRatio":1, "rebin":rebin}

        plot_args["outputFolder"] = args.outputFolder
    
        fig = makePlot(cfg, **plot_args)
        plt.close()




    

if __name__ == '__main__':

    args = parse_args()

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

    doPlots(debug=args.debug)
