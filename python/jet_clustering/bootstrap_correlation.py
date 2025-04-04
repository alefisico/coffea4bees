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
import awkward as ak
import yaml
import copy

sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.helpers_make_plot_dict as plot_helpers_make_plot_dict
import base_class.plots.helpers_make_plot as plot_helpers_make_plot
import base_class.plots.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')

def add_to_dict(event_id, SvB, SvB_per_event):
    if event_id not in SvB_per_event:
        SvB_per_event[event_id] = []

    SvB_per_event[event_id].append(SvB)


def main():

    input_file = cfg.hists[0]

    # 'event', 'run', 'luminosityBlock', 'SvB_MA_ps'
    vX = [f"v{i}" for i in range(16)]
    eras = ["2022_preEE","2022_EE", "2023_preBPix", "2023_BPix"]

    #
    #  Start with 2 vs and one era
    #
    #vX   = [f"v{i}" for i in range(16)]
    #eras = ["2022_preEE"]

    #
    # combine data by event/run/LB
    #
    SvB_per_event = {}
    SvB_vX = {}
    for _vX in vX:
        SvB_vX[_vX] = []

        for _era in eras:

            _key = f"syn_{_vX}_{_era}"
            #_key = f"syn_v0_{_era}"
            print(_key)

            SvB_vX[_vX] += input_file["SvB_MA_ps"][_key]

            nEvents = len(input_file["event"][_key])
            for _iE in range(nEvents):
                SvB    = input_file["SvB_MA_ps"]      [_key][_iE]
                if SvB < 0.001: continue

                event  = input_file["event"]          [_key][_iE]
                run    = input_file["run"]            [_key][_iE]
                LB     = input_file["luminosityBlock"][_key][_iE]

                event_id = (event, run, LB)
                add_to_dict(event_id, SvB, SvB_per_event)


    #
    #  Add poission weights
    #
    N_events = len(SvB_per_event)
    N_toy = 30
    weights = np.random.poisson(lam=1, size=(N_events, N_toy))

    #
    #  Get SvB Values per event
    #
    SvB_values = ak.Array(list(SvB_per_event.values()))
    count_SvB_values = ak.num(SvB_values)

    #
    #  Compute the Average and the bins
    #
    counts_tot, bin_edges = np.histogram( ak.flatten(SvB_values), bins=40)
    counts_ave = counts_tot / len(vX)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (bin_centers[1] - bin_centers[0])
    niave_error = np.sqrt(counts_tot)/16

    #
    #  Do the bootstrapping
    #
    bootstrap_counts = np.zeros((N_toy, len(counts_tot)))
    for i in range(N_toy):
        weights_expanded = np.array([w for w, _count in zip(weights[:,i], count_SvB_values) for _ in range(_count)])
        weights_ave = weights_expanded * 1./len(vX)
        bootstrap_counts[i], _ = np.histogram( ak.flatten(SvB_values).to_numpy(), bins=bin_edges, weights=weights_ave)

    bootstrap_rms = np.sqrt(np.mean((bootstrap_counts - counts_ave)**2, axis=0))

    #
    #  Counts
    #
    counts_vX = []
    diff_wrt_ave_vX = []
    for _vX in vX:

        _SvB = np.array(SvB_vX[_vX])
        _SvB = _SvB[_SvB > 0.001]
        _counts_vX, _ = np.histogram( _SvB, bins=bin_edges)
        counts_vX.append(_counts_vX)

        diff_wrt_ave_vX.append(_counts_vX - counts_ave)

    np.mean(counts_vX, axis=0)
    np.mean(diff_wrt_ave_vX, axis=0) # sould be all zero
    var_vX_wrt_ave = np.mean(np.square(diff_wrt_ave_vX), axis=0)
    rms_wrt_ave = np.sqrt(var_vX_wrt_ave)


    #
    #  Plots Dictionaries
    #
    default_dict = {"tag": "fourTag", "centers": bin_centers, "edges": bin_edges, "x_label":"SvB", "under_flow":0, "over_flow":0, "edgecolor": "k", "fillcolor": "k"}

    data_v0_dict = copy.deepcopy(default_dict)
    data_v0_dict["label"] = "One Synthetic Dataset"
    data_v0_dict["values"] = counts_vX[0]
    data_v0_dict["variances"] = counts_vX[0]

    data_v1_dict = copy.deepcopy(default_dict)
    data_v1_dict["label"] = "v0"
    data_v1_dict["edgecolor"] = "r"
    data_v1_dict["fillcolor"] = "r"
    data_v1_dict["values"] = counts_vX[1]
    data_v1_dict["variances"] = counts_vX[1]

    ave_dict = copy.deepcopy(default_dict)
    ave_dict["label"] = "Average (Bootstrap Uncertainties)"
    ave_dict["edgecolor"] = "k"
    ave_dict["fillcolor"] = "#FFDF7Fff"
    ave_dict["values"] = counts_ave
    ave_dict["variances"] = bootstrap_rms * bootstrap_rms

    ave_dict_niave = copy.deepcopy(default_dict)
    ave_dict_niave["label"] = "Average (Naive Uncertainties)"
    ave_dict_niave["edgecolor"] = "k"
    ave_dict_niave["fillcolor"] = "#FFDF7Fff"
    ave_dict_niave["values"] = counts_ave
    ave_dict_niave["variances"] = niave_error * niave_error


    #
    #  Plots  vX vs Ave
    #
    plot_data_one = {}
    plot_data_one["hists"] = {"v0": data_v0_dict,
                              #"v1": data_v1_dict,
                              }
    plot_data_one["stack"] = {"ave": ave_dict_niave}
    plot_data_one["ratio"] = {}
    plot_data_one["region"] = "SR"
    plot_data_one["cut"] = "one_vs_ave"
    plot_data_one["var"] = "SvB"
    plot_data_one["kwargs"] = {"debug":True, "doratio":False, "outputFolder":"./", "yscale":"log", "rlim":[0.5,1.5], "year":"Run3"}

    ratio_config = {"v0_to_ave":
                    {"numerator": {"type": "hists", "key": "v0"}, "denominator": {"type": "stack"},
                     "uncertianty": "nominal", "color":"k", "marker":"o"}}
    plot_helpers_make_plot_dict.add_ratio_plots(ratio_config, plot_data_one, *{})
    plot_helpers_make_plot.make_plot_from_dict(plot_data_one)


    #
    #  Plots  Niave Ave  vs Ave
    #
    plot_data_ave = {}
    ave_dict_niave["fillcolor"] = "k"
    plot_data_ave["hists"] = {"niave": ave_dict_niave,
                              }
    plot_data_ave["stack"] = {"ave": ave_dict}
    plot_data_ave["ratio"] = {}
    plot_data_ave["region"] = "SR"
    plot_data_ave["cut"] = "ave_vs_ave"
    plot_data_ave["var"] = "SvB"
    plot_data_ave["kwargs"] = {"debug":True, "doratio":False, "outputFolder":"./", "yscale":"log", "rlim":[0.9,1.1], "year":"Run3"}

    ratio_config = {"niave_to_ave":
                    {"numerator": {"type": "hists", "key": "niave"}, "denominator": {"type": "stack"},
                     "uncertianty": "nominal", "color":"k", "marker":"o", "bkg_err_band": None},
                    }

    default_band_config = {"color": "k",  "type": "band", "hatch": "\\\\\\"}
    #vX_var_band_config = copy.deepcopy(default_band_config)
    ##vX_var_band_config["hatch"] = None
    ##vX_var_band_config["facecolor"] = "#FFDF7Fff"
    #vX_var_band_config["ratio"] = np.ones(len(bin_centers)).tolist()
    #vX_var_band_config["error"] = np.sqrt(var_vX_wrt_ave * np.power(ave_dict["values"], -2.0)).tolist()
    #vX_var_band_config["centers"] = bin_centers
    #plot_data_ave["ratio"]["band_var"] = vX_var_band_config



    band_config = copy.deepcopy(default_band_config)
    band_config["hatch"] = None
    band_config["facecolor"] = "#FFDF7Fff"
    band_config["ratio"] = np.ones(len(bin_centers)).tolist()
    #ave_dict["values"][ave_dict["values"] == 0] = plot_helpers.epsilon
    band_config["error"] = np.sqrt(ave_dict["variances"] * np.power(ave_dict["values"], -2.0)).tolist()
    band_config["centers"] = bin_centers
    plot_data_ave["ratio"]["band_boot"] = band_config

    plot_helpers_make_plot_dict.add_ratio_plots(ratio_config, plot_data_ave, *{})

    plot_helpers_make_plot.make_plot_from_dict(plot_data_ave)




#    plot_data_one = {}
#    plot_data_one["hists"] = {"v0": data_v0_dict,
#                              #"v1": data_v1_dict,
#                              }
#    plot_data_one["stack"] = {"ave": ave_dict}
#    plot_data_one["ratio"] = {}
#    plot_data_one["region"] = "SR"
#    plot_data_one["cut"] = "passPreSel"
#    plot_data_one["var"] = "SvB"
#    plot_data_one["kwargs"] = {"debug":True, "doratio":False, "outputFolder":"./", "yscale":"log"}
#



    #breakpoint()
    #counts_ave
    #bootstrap_counts[i], _ = np.histogram(original_df['total_pt'], bins=bin_edges, weights=original_df[f'weight_{i}'])





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

    main()
