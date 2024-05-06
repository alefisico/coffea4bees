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

    if args.doTest:
        varList = ["SvB_MA.ps_zz", "SvB_MA.ps_zh", "SvB_MA.ps_hh", "quadJet_selected.lead_vs_subl_m", "quadJet_min_dr.close_vs_other_m"]

    #
    #  Nominal 1D Plots
    #
    for v in varList:
        if debug: print(f"plotting 1D ...{v}")

        vDict = cfg.plotModifiers.get(v, {})
        print(v, vDict, vDict.get("2d", False))
        if vDict.get("2d", False):
            continue

        cut = "passPreSel"
        tag = "fourTag"

        vDict["ylabel"] = "Entries"
        vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
        vDict["legend"] = True

        for region in ["SR", "SB"]:

            if debug: print(f"plotting 1D ...{v}")
            plot_args  = {}
            plot_args["var"] = v
            plot_args["cut"] = cut
            plot_args["region"] = region
            plot_args["outputFolder"] = args.outputFolder
            plot_args = plot_args | vDict
            if debug: print(plot_args)
            fig = makePlot(cfg, **plot_args)

            plt.close()

    #
    #  2D Plots
    #
    for v in varList:
        print(v)

        vDict = cfg.plotModifiers.get(v, {})

        if not vDict.get("2d", False):
            continue

        vDict["ylabel"] = "Entries"
        vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
        vDict["legend"] = True

        for process in ["data", "Multijet", "HH4b", "TTToHadronic"]:
            for region in ["SR", "SB"]:

                plot_args  = {}
                plot_args["var"] = v
                plot_args["cut"] = cut
                plot_args["region"] = region
                plot_args["outputFolder"] = args.outputFolder
                plot_args = plot_args | vDict

                if debug: print("process is ",process)
                if debug: print(plot_args)

                fig = make2DPlot(cfg, process,
                                 **plot_args)
                plt.close()

    #
    #  Comparison Plots
    #
    varListComp = []
    if args.doTest:
        varListComp = ["v4j.mass", "SvB_MA.ps", "quadJet_selected.xHH"]

        for v in varListComp:
            print(v)

            vDict = cfg.plotModifiers.get(v, {})

            vDict["ylabel"] = "Entries"
            vDict["doRatio"] = cfg.plotConfig.get("doRatio", True)
            vDict["legend"] = True

            for process in ["data", "Multijet", "HH4b", "TTToHadronic"]:

                #
                # Comp Cuts
                #
                for region in ["SR", "SB"]:

                    plot_args  = {}
                    plot_args["var"] = v
                    plot_args["cut"] = ["passPreSel", "failSvB", "passSvB"]
                    plot_args["region"] = region
                    plot_args["outputFolder"] = args.outputFolder
                    plot_args["process"] = process
                    plot_args["norm"] = True
                    plot_args = plot_args | vDict

                    if debug: print(plot_args)

                    fig = makePlot(cfg, **plot_args)



                    plt.close()

                #
                # Comp Regions
                #
                fig = makePlot(cfg,
                               var=v,
                               cut="passPreSel",
                               region=["SR", "SB"],
                               process=process,
                               outputFolder=args.outputFolder,
                               **vDict
                               )

                plt.close()


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

    varList = [ h for h in cfg.hists[0]['hists'].keys() if not h in args.skip_hists ]
    doPlots(varList, debug=args.debug)
