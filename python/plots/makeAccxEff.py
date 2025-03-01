# From
# https://github.com/patrickbryant/ZZ4b/blob/master/nTupleAnalysis/scripts/makeAccxEff.py
#


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
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl

sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.iPlot_config as cfg
import base_class.plots.helpers as plot_helpers

np.seterr(divide='ignore', invalid='ignore')


colors = ["DFDAFDSADS",
          "xkcd:light purple",
          "blue",
          "green",
          "red",
          "black",
          "xkcd:light purple"]


cuts_flow = [("passCleanGenWeight",0, "Denominator"),
             ("passJetMult",0, "$\geq4$ selected jets" ), # purple
             ("passPreSel_woTrig",1, "4 b-tagged jets"),  # blue
             ("passDiJetMass_woTrig",1, "m(j,j)"), # green
             ("SR_woTrig",1 , "Signal region"),     #
             ("SR",1, "Trigger")
             ]



def _get_hist_data(input_hist):
    config = {}
    config["values"]     = input_hist.values().tolist()
    #config["variances"]  = input_hist.variances().tolist()
    config["variances"]  = input_hist.values().tolist()
    config["centers"]    = input_hist.axes[0].centers.tolist()
    config["edges"]      = input_hist.axes[0].edges.tolist()
    config["x_label"]    = input_hist.axes[0].label
    #config["under_flow"] = float(input_hist.view(flow=True)["value"][0])
    #config["over_flow"]  = float(input_hist.view(flow=True)["value"][-1])
    return config


def get_hist_data(in_file, hist_name, hist_key, rebin):
    input_hist = in_file['cutflow_hists'][hist_key][hist_name]
    input_hist = input_hist[::hist.rebin(rebin)]
    return _get_hist_data(input_hist)


def makeEffPlot(name, data_to_plot, cuts_flow, **kwargs):
    size = 7
    fig = plt.figure()   # figsize=(size,size/_phi))
    fig.add_axes((0.1, 0.15, 0.85, 0.8))
    main_ax = fig.gca()
    ratio_ax = None

    year_str = plot_helpers.get_year_str(kwargs.get("year","Run2"))

    hep.cms.label("Internal", data=False,
                  year=year_str, loc=0, ax=main_ax)


    for ic in range(1,len(cuts_flow)):
        cut_name = cuts_flow[ic][0]
        plt.plot(data_to_plot[cut_name]["centers"], data_to_plot[cut_name]["ratio"], marker='o', linestyle='-', color=colors[ic], label=cuts_flow[ic][2])

        #plt.errorbar(
        #    data_to_plot[cut_name]["centers"],
        #    data_to_plot[cut_name]["ratio"],
        #    yerr = np.sqrt(data_to_plot[cut_name]["error"]),
        #    fmt='o',             # marker style (same as marker='o')
        #    color=colors[ic],
        #    linestyle='-',       # connect points with a line
        #    ecolor='gray',  # color for error bars
        #    capsize=2            # length of the caps on the error bars
        #)

    plt.xlim(300,1200)
    plt.ylim(kwargs.get("ylim",[0,1.3]))
    plt.yscale(kwargs.get("yscale","linear"))
    plt.xlabel("$m_{4b}^{gen}$ [GeV]")
    plt.ylabel("Acceptance x Efficiency")
    plt.legend(loc="upper left", ncol=2,
               bbox_to_anchor=(0.025, .975), # Moves the legend outside and centers it
               fontsize = "large"
               )


    plt.savefig(f"{name}_{year_str}.pdf")





def makePlot(cfg, year, debug=False):


    process = "GluGluToHHTo4B_cHHH1"
    hist_key = f"{process}_{year}"

    rebin = 10

    tot_eff = {}
    rel_eff = {}

    den_tot_data = get_hist_data(cfg.hists[0], cuts_flow[0][0], hist_key, rebin)

    #
    # Compute the SF between two input files
    #
    data_0 = get_hist_data(cfg.hists[0], "SR_woTrig", hist_key, rebin)
    data_1 = get_hist_data(cfg.hists[1], "SR_woTrig", hist_key, rebin)

    print(f"\tnorm file0 {np.sum(data_0['values'])} vs file1: {np.sum(data_1['values'])} ratio {np.sum(data_0['values'])/np.sum(data_1['values'])}")

    scalefactor = np.sum(data_0['values'])/np.sum(data_1['values'])

    for ic in range(1,len(cuts_flow)):

        print(ic, cuts_flow[ic])

        den_cut_name = cuts_flow[ic - 1][0]
        den_file_idx = cuts_flow[ic - 1][1]
        den_hist = cfg.hists[den_file_idx]['cutflow_hists'][hist_key][den_cut_name]
        #den_hist = den_hist.rebin(rebin)
        den_hist = den_hist[::hist.rebin(rebin)]
        den_data = _get_hist_data(den_hist)

        num_cut_name = cuts_flow[ic][0]
        num_file_idx = cuts_flow[ic][1]
        num_hist = cfg.hists[num_file_idx]['cutflow_hists'][hist_key][num_cut_name]
        #num_hist = num_hist.rebin(rebin)
        num_hist = num_hist[::hist.rebin(rebin)]
        num_data = _get_hist_data(num_hist)


        thisSF = 1.0
        if not den_file_idx == num_file_idx:
            thisSF = scalefactor

        ratios, ratio_uncert = plot_helpers.makeRatio(np.array(num_data["values"])*thisSF,
                                                      np.array(num_data["variances"]),
                                                      np.array(den_data["values"]),
                                                      np.array(den_data["variances"]))
        rel_eff[cuts_flow[ic][0]] = {"ratio":ratios, "error":ratio_uncert, "centers":num_data["centers"]}

        if not num_file_idx == 0:
            thisSF = scalefactor

        ratios_tot, ratio_tot_uncert = plot_helpers.makeRatio(np.array(num_data["values"])*thisSF,
                                                              np.array(num_data["variances"]),
                                                              np.array(den_tot_data["values"]),
                                                              np.array(den_tot_data["variances"]))
        tot_eff[cuts_flow[ic][0]] = {"ratio":ratios_tot, "error":ratio_tot_uncert, "centers":num_data["centers"]}





    #
    #
    #
    makeEffPlot("total_eff",    tot_eff, cuts_flow, yscale="log", year=year, ylim=[1e-3, 10])
    makeEffPlot("relative_eff", rel_eff, cuts_flow, year=year)


    return




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


    for y in ["UL18", "UL17","UL16_preVFP", "UL16_postVFP"]: #,"RunII"]:
        makePlot(cfg, year=y, debug=args.debug)
