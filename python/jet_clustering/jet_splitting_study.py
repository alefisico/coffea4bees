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
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args, get_cut_dict
import base_class.plots.iPlot_config as cfg
from make_jet_splitting_PDFs import make_PDFs_vs_Pt

np.seterr(divide='ignore', invalid='ignore')

def plot(var, **kwargs):
    fig, ax = makePlot(cfg, var, outputFolder= args.outputFolder, **kwargs)
    plt.close()
    return fig, ax


def plot2d(process, **kwargs):
    fig, ax = make2DPlot(cfg, process, outputFolder= args.outputFolder, **kwargs)
    plt.close()
    return fig, ax


def write_1D_pdf(output_file, varName, bin_centers, probs, n_spaces=4):
    spaces = " " * n_spaces
    output_file.write(f"{spaces}{varName}:\n")
    output_file.write(f"{spaces}    bin_centers:  {bin_centers.tolist()}\n")
    output_file.write(f"{spaces}    probs:  {probs.tolist()}\n")


def write_2D_pdf(output_file, varName, hist, n_spaces=4):

    counts = hist.view(flow=False)

    xedges = hist.axes[0].edges
    yedges = hist.axes[1].edges
    probabilities = counts.value / counts.value.sum()

    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    probabilities_flat = probabilities.flatten()

    spaces = " " * n_spaces
    output_file.write(f"{spaces}{varName}:\n")
    output_file.write(f"{spaces}    xcenters:  {xcenters.tolist()}\n")
    output_file.write(f"{spaces}    ycenters:  {ycenters.tolist()}\n")
    output_file.write(f"{spaces}    probabilities_flat:  {probabilities_flat.tolist()}\n")


pt_names = {0: "$p_{T}$: < 140 GeV",
            1: "$p_{T}$: 140 - 230",
            2: "$p_{T}$: 230 - 320",
            3: "$p_{T}$: 320 - 410",
            4: "$p_{T}$: > 410 GeV",
            }


eta_names = {0: "$\eta_{T}$: < 0.6",
             1: "$\eta_{T}$: 0.6 - 1.2",
             2: "$\eta_{T}$: 1.2 - 1.8",
             3: "$\eta_{T}$: 1.8 - 2.4",
             4: "$\eta_{T}$: > 2.4",
            }






def centers_to_edges(centers):
    bin_width = centers[1] - centers[0]

    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = (centers[1:] + centers[:-1]) / 2
    edges[0] = centers[0] - bin_width / 2
    edges[-1] = centers[-1] + bin_width / 2
    return edges


def get_bins_xMin_xMax_from_centers(centers):
    nBins = len(centers)
    bin_half_width = 0.5*(centers[1]  - centers[0])
    xMin  = centers[0]  - bin_half_width
    xMax  = centers[-1] + bin_half_width

    return nBins, xMin, xMax




def doPlots(debug=False):

    #
    #  config Setup
    #
    splitting_config = {}

    s_XX     = { "mA":("mA",   1),  "mB":("mB",   1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_vs_thetaA", 1) }
    s_XX_X   = { "mA":("mA_l", 1),  "mB":("mB",   1), "decay_phi":("decay_phi", 4), "zA_vs_thetaA":("zA_vs_thetaA", 1) }

    splitting_config["bb"]    = s_XX
    splitting_config["bj"]    = s_XX
    splitting_config["jj"]    = s_XX

    splitting_config["(bj)b"] = s_XX_X

    splittings = list(splitting_config.keys())
    varNames   = list(splitting_config[splittings[0]].keys())

    output_file_name_vs_pT = args.outputFolder+"/clustering_pdfs_vs_pT.yml"
    make_PDFs_vs_Pt(splitting_config, output_file_name_vs_pT)

    pt_bins = [0, 140, 230, 320, 410, np.inf]

    with open(f'{output_file_name_vs_pT}', 'r') as output_file_vs_pT:

        for _s in splittings:

            for _v in varNames:
                _hist_name = f"splitting_{_s}.{_v}_pT"

                if _v.find("_vs_") == -1:
                    is_1d_hist = True
                    plt.figure(figsize=(6, 6))
                else:
                    is_1d_hist = False
                    plt.figure(figsize=(18, 12))

                for _iPt in range(len(pt_bins) - 1):

                    cut_dict = get_cut_dict("passPreSel", cfg.cutList)
                    plot_dict = {"process":"data", "year":sum, "tag":1,"region":sum,"pt":_iPt}
                    plot_dict = plot_dict | cut_dict

                    if is_1d_hist:
                        cfg.hists[0]["hists"][_hist_name][plot_dict].plot(label=f"{pt_names[_iPt]}")
                    else:
                        plt.subplot(2, 3, _iPt + 1)
                        cfg.hists[0]["hists"][_hist_name][plot_dict].plot2d()
                        plt.title(f'{pt_names[_iPt]}')
                plt.legend()
                plt.savefig(args.outputFolder+f"/test_pt_dependence_{_s}_{_v}.pdf")



    #
    #  Plots vs Eta
    #
    for _s in splittings:

        for _v in varNames:
            _hist_name = f"splitting_{_s}.{_v}_eta"

            if _v.find("_vs_") == -1:
                is_1d_hist = True
                plt.figure(figsize=(6, 6))
            else:
                is_1d_hist = False
                plt.figure(figsize=(18, 12))

            for _iPt in range(len(pt_bins) - 1):

                cut_dict = get_cut_dict("passPreSel", cfg.cutList)
                plot_dict = {"process":"data", "year":sum, "tag":1,"region":sum, "abs_eta":_iPt}
                plot_dict = plot_dict | cut_dict

                if is_1d_hist:
                    cfg.hists[0]["hists"][_hist_name][plot_dict].plot(label=f"{eta_names[_iPt]}")
                else:
                    plt.subplot(2, 3, _iPt + 1)
                    cfg.hists[0]["hists"][_hist_name][plot_dict].plot2d()
                    plt.title(f'{eta_names[_iPt]}')
            plt.legend()
            plt.savefig(args.outputFolder+f"/test_eta_dependence_{_s}_{_v}.pdf")



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
