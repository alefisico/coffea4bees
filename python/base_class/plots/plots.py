import os
import sys
import hist
import yaml
import copy
import argparse
from coffea.util import load
from hist.intervals import ratio_uncertainty
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
plt.style.use([hep.style.CMS, {'font.size': 16}])
import inspect
import base_class.plots.iPlot_config as cfg
from collections import defaultdict

_phi = (1 + np.sqrt(5)) / 2
_epsilon = 0.001
_colors = ["xkcd:black", "xkcd:red", "xkcd:off green", "xkcd:blue",
           "xkcd:orange", "xkcd:violet", "xkcd:grey",
           "xkcd:pink" , "xkcd:pale blue"]


def load_config(metadata):
    """  Load meta data
    """
    plotConfig = yaml.safe_load(open(metadata, 'r'))

    #
    #  Make two way code mapping:
    #    ie: 3 mapts to  "threeTag" and "threeTag" maps to 3
    for k, v in plotConfig["codes"]["tag"].copy().items():
        plotConfig["codes"]["tag"][v] = k

    for k, v in plotConfig["codes"]["region"].copy().items():
        if type(v) is list:
            continue
        plotConfig["codes"]["region"][v] = k

    return plotConfig


def get_value_nested_dict(nested_dict, target_key):
    """ Return the first value from mathching key from nested dict
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v


        if type(v) is dict:
            try:
                return get_value_nested_dict(v, target_key)
            except ValueError:
                continue

    raise ValueError(f"\t target_key {target_key} not in nested_dict")

def get_hist_data(hist):
    values    = hist.values().tolist()
    variances = hist.variances().tolist()
    centers   = hist.axes[0].centers.tolist()
    edges     = hist.axes[0].edges.tolist()
    label     = hist.axes[0].label

    return values, variances, edges, centers, label

def get_values_variances_centers_from_dict(hist_config, hists, stack_dict):

    if hist_config["type"] == "hists":
        for h_data, h_config in hists:
            if h_config["name"] == hist_config["key"]:
                return h_data.values(), h_data.variances(), h_data.axes[0].centers

        raise ValueError(f"ERROR: input to ratio of key {hist_config['key']} not found in hists")


    if hist_config["type"] == "stack":
        stack_dict_for_hist = {k: v[0] for k, v in stack_dict.items() }
        hStackHists = list(stack_dict_for_hist.values())
        return_values = [h.values() for h in hStackHists]
        return_values = np.sum(return_values, axis=0)

        return_variances = [h.variances() for h in hStackHists]
        return_variances = np.sum(return_variances, axis=0)

        return return_values, return_variances, hStackHists[0].axes[0].centers

    raise ValueError("ERROR: ratio needs to be of type 'hists' or 'stack'")


def init_config(args):
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

    return cfg

def _savefig(fig, var, *args):

    args_str = []
    for _arg in args:
        if type(_arg) is list:
            args_str.append( "_vs_".join(_arg) )
        else:
            args_str.append(_arg)

    outputPath = "/".join(args_str)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    varStr = var if type(var) is str else "_vs_".join(var)
    fig.savefig(outputPath + "/" + varStr.replace(".", '_').replace("/","_") + ".pdf")
    return


def get_cut_dict(cut, cutList):
    cutDict = {}
    for c in cutList:
        cutDict[c] = sum
    cutDict[cut] = True
    return cutDict


def print_list_debug_info(process, tag, cut, region):
    print(f" hist process={process}, "
          f"tag={tag}, _cut={cut}"
          f"_reg={region}")


def get_label(default_str, override_list, i):
    return override_list[i] if (override_list and len(override_list) > i) else default_str



#
#  Get hist from input file(s)
#
def get_hist(cfg, config, var, region, cut, rebin, year, file_index=None, debug=False):

    codes = cfg.plotConfig["codes"]

    if year == "RunII":
        year     = sum

    tag_code = codes["tag"][config["tag"]]

    if debug:
        print(f" hist process={config['process']}, "
              f"tag={tag_code}, year={year}, var={var}")

    hist_opts = {"process": config['process'],
                 "year":  year,
                 "tag":   hist.loc(tag_code),
                 }


    if (not region  == "sum") and (type(codes["region"][region]) is list):
        region_dict = {"region":  [hist.loc(r) for r in codes["region"][region]]}
    else:
        if region == "sum":
            region_dict = {"region":  sum}
        else:
            region_dict = {"region":  hist.loc(codes["region"][region])}

    cut_dict = get_cut_dict(cut, cfg.cutList)

    hist_opts = hist_opts | region_dict | cut_dict

    hist_obj = None
    if len(cfg.hists) > 1 and not cfg.combine_input_files:
        if file_index is None:
            print("ERROR must give file_index if running with more than one input file without using the  --combine_input_files option")
        hist_obj = cfg.hists[file_index]['hists'][var]

    else:
        for _input_data in cfg.hists:
            if var in _input_data['hists'] and config['process'] in _input_data['hists'][var].axes["process"]:
                hist_obj = _input_data['hists'][var]

    if hist_obj is None:
        raise ValueError(f"ERROR did not find var {var} with process {config['process']} in inputs")

    #
    #  Add rebin Options
    #
    varName = hist_obj.axes[-1].name
    var_dict = {varName: hist.rebin(rebin)}
    hist_opts = hist_opts | var_dict

    #
    #  Do the hist selection/binngin
    #
    selected_hist = hist_obj[hist_opts]

    #
    # Catch list vs hist
    #  Shape give (nregion, nBins)
    #
    if len(selected_hist.shape) == 2:
        selected_hist = selected_hist[sum, :]

    #
    # Apply Scale factor
    #
    selected_hist *= config.get("scalefactor", 1.0)

    return selected_hist


def makeRatio(numValues, numVars, denValues, denVars, **kwargs):

    denValues[denValues == 0] = _epsilon
    ratios = numValues / denValues

    if kwargs.get("norm", False):
        numSF = np.sum(numValues, axis=0)
        denSF = np.sum(denValues, axis=0)
        ratios *= denSF / numSF

    # Set 0 and inf to nan to hide during plotting
    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan

    # Both num and denom. uncertianties
    # ratio_uncert = np.abs(ratios) * np.sqrt(numVars * np.power(numValues, -2.0) + denVars * np.power(denValues, -2.0 ))
    ratio_uncert = np.abs(ratios) * np.sqrt(numVars * np.power(numValues, -2.0))

    ## https://github.com/scikit-hep/hist/blob/main/src/hist/intervals.py
    #ratio_uncert = ratio_uncertainty(
    #    num=numValues,
    #    denom=denValues,
    #    uncertainty_type=kwargs.get("uncertainty_type", "poisson"),
    #)

    return ratios, ratio_uncert


def _draw_plot(hist_list, stack_dict, **kwargs):
    r"""
    Takes options:
          "norm"   : bool
          "debug"  : bool
          "xlabel" : string
          "ylabel" : string
          "yscale" : 'log' | None
          "xscale" : 'log' | None
          "legend" : bool
          'ylim'   : [min, max]
          'xlim'   : [min, max]
    """

    if kwargs.get("debug", False):
        print(f'\t in _draw_plot ... kwargs = {kwargs}')
    norm = kwargs.get("norm", False)

    #
    #  Draw the stack
    #
    stack_dict_for_hist = {k: v[0] for k, v in stack_dict.items() }
    stack_colors_fill   = [ v[1].get("fillcolor") for _, v in stack_dict.items() ]
    stack_colors_edge   = [ v[1].get("edgecolor") for _, v in stack_dict.items() ]

    if len(stack_dict_for_hist):
        s = hist.Stack.from_dict(stack_dict_for_hist)

        s.plot(stack=True, histtype="fill",
               color=stack_colors_fill,
               label=None,
               density=norm)

        s.plot(stack=True, histtype="step",
               color=stack_colors_edge,
               label=None,
               density=norm)

    stack_patches = []

    #
    # Add the stack components to the legend
    #
    for proc_name, proc_data in stack_dict.items():
        proc_config = proc_data[1]
        _label = proc_config.get('label')

        if _label in ["None"]:
            continue

        stack_patches.append(mpatches.Patch(facecolor=proc_config.get("fillcolor"),
                                            edgecolor=proc_config.get("edgecolor"),
                                            label=_label))

    #
    #  Draw the hists
    #
    hist_artists = []

    for hist_data in hist_list:
        hist_obj    = hist_data[0]
        hist_config = hist_data[1]
        _plot_options = {"density":  norm,
                         "label":    hist_config.get("label", ""),
                         "color":    hist_config.get('fillcolor', 'k'),
                         "histtype": kwargs.get("histtype", hist_config.get("histtype", "errorbar")),
                         "linewidth": kwargs.get("linewidth", hist_config.get("linewidth", 1)),
                         "yerr": False,
                         }

        if kwargs.get("histtype", hist_config.get("histtype", "errorbar")) in ["errorbar"]:
            _plot_options["markersize"] = 12
            _plot_options["yerr"] = True

        hist_artists.append(hist_obj.plot(**_plot_options)[0])

    #
    #  xlabel
    #
    if kwargs.get("xlabel", None):
        plt.xlabel(kwargs.get("xlabel"))
    plt.xlabel(plt.gca().get_xlabel(), loc='right')

    #
    #  ylabel
    #
    if kwargs.get("ylabel", None):
        plt.ylabel(kwargs.get("ylabel"))
    if norm:
        plt.ylabel(plt.gca().get_ylabel() + " (normalized)")
    plt.ylabel(plt.gca().get_ylabel(), loc='top')

    if kwargs.get("yscale", None):
        plt.yscale(kwargs.get('yscale'))
    if kwargs.get("xscale", None):
        plt.xscale(kwargs.get('xscale'))

    if kwargs.get('legend', True):
        handles, labels = plt.gca().get_legend_handles_labels()

        for s in stack_patches:
            handles.append(s)
            labels.append(s.get_label())

        plt.legend(
            handles=handles,
            labels=labels,
            loc='best',      # Position of the legend
            # fontsize='medium',      # Font size of the legend text
            frameon=False,           # Display frame around legend
            # framealpha=0.0,         # Opacity of frame (1.0 is opaque)
            # edgecolor='black',      # Color of the legend frame
            # title='Trigonometric Functions',  # Title for the legend
            # title_fontsize='large',# Font size of the legend title
            # bbox_to_anchor=(1, 1), # Specify position of legend
            # borderaxespad=0.0 # Padding between axes and legend border
            reverse=True,
        )

    if kwargs.get('ylim', False):
        plt.ylim(*kwargs.get('ylim'))
    if kwargs.get('xlim', False):
        plt.xlim(*kwargs.get('xlim'))

    return

def _draw_plot_from_dict(plot_data, **kwargs):
    r"""
    Takes options:
          "norm"   : bool
          "debug"  : bool
          "xlabel" : string
          "ylabel" : string
          "yscale" : 'log' | None
          "xscale" : 'log' | None
          "legend" : bool
          'ylim'   : [min, max]
          'xlim'   : [min, max]
    """

    if kwargs.get("debug", False):
        print(f'\t in _draw_plot ... kwargs = {kwargs}')
    norm = kwargs.get("norm", False)

    #
    #  Draw the stack
    #
    stack_dict = plot_data["stack"]
    stack_dict_for_hist = {k: v[0] for k, v in stack_dict.items() }
    stack_colors_fill   = [ v[1].get("fillcolor") for _, v in stack_dict.items() ]
    stack_colors_edge   = [ v[1].get("edgecolor") for _, v in stack_dict.items() ]

    if len(stack_dict_for_hist):
        s = hist.Stack.from_dict(stack_dict_for_hist)

        s.plot(stack=True, histtype="fill",
               color=stack_colors_fill,
               label=None,
               density=norm)

        s.plot(stack=True, histtype="step",
               color=stack_colors_edge,
               label=None,
               density=norm)

    stack_patches = []

    #
    # Add the stack components to the legend
    #
    for proc_name, proc_data in stack_dict.items():
        proc_config = proc_data[1]
        _label = proc_config.get('label')

        if _label in ["None"]:
            continue

        stack_patches.append(mpatches.Patch(facecolor=proc_config.get("fillcolor"),
                                            edgecolor=proc_config.get("edgecolor"),
                                            label=_label))

    #
    #  Draw the hists
    #
    hist_artists = []
    hist_objs = []
    for hist_proc_name, hist_data in plot_data["hists"].items():
        #hist_obj    = hist_data[0]

        hist_obj = hist.Hist(
            hist.axis.Variable(hist_data["edges"], name=hist_data["x_label"]),  # Define variable-width bins
            storage=hist.storage.Weight()           # Use Weight storage for counts and variances
        )
        hist_obj[...] = np.array(list(zip(hist_data["values"], hist_data["variance"])), dtype=[("value", "f8"), ("variance", "f8")])

        print("plotting", hist_proc_name)

        _plot_options = {"density":  norm,
                         "label":    hist_data.get("label", ""),
                         "color":    hist_data.get('fillcolor', 'k'),
                         "histtype": kwargs.get("histtype", hist_data.get("histtype", "errorbar")),
                         "linewidth": kwargs.get("linewidth", hist_data.get("linewidth", 1)),
                         "yerr": False,
                         }

        if kwargs.get("histtype", hist_data.get("histtype", "errorbar")) in ["errorbar"]:
            _plot_options["markersize"] = 12
            _plot_options["yerr"] = True
        hist_artists.append(hist_obj.plot(**_plot_options)[0])

    #
    #  xlabel
    #
    if kwargs.get("xlabel", None):
        plt.xlabel(kwargs.get("xlabel"))
    plt.xlabel(plt.gca().get_xlabel(), loc='right')

    #
    #  ylabel
    #
    if kwargs.get("ylabel", None):
        plt.ylabel(kwargs.get("ylabel"))
    if norm:
        plt.ylabel(plt.gca().get_ylabel() + " (normalized)")
    plt.ylabel(plt.gca().get_ylabel(), loc='top')

    if kwargs.get("yscale", None):
        plt.yscale(kwargs.get('yscale'))
    if kwargs.get("xscale", None):
        plt.xscale(kwargs.get('xscale'))

    if kwargs.get('legend', True):
        handles, labels = plt.gca().get_legend_handles_labels()

        for s in stack_patches:
            handles.append(s)
            labels.append(s.get_label())

        plt.legend(
            handles=handles,
            labels=labels,
            loc='best',      # Position of the legend
            # fontsize='medium',      # Font size of the legend text
            frameon=False,           # Display frame around legend
            # framealpha=0.0,         # Opacity of frame (1.0 is opaque)
            # edgecolor='black',      # Color of the legend frame
            # title='Trigonometric Functions',  # Title for the legend
            # title_fontsize='large',# Font size of the legend title
            # bbox_to_anchor=(1, 1), # Specify position of legend
            # borderaxespad=0.0 # Padding between axes and legend border
            reverse=True,
        )

    if kwargs.get('ylim', False):
        plt.ylim(*kwargs.get('ylim'))
    if kwargs.get('xlim', False):
        plt.xlim(*kwargs.get('xlim'))

    return


def get_year_str(year):

    if type(year) is list:
        year_str = "_vs_".join(year)
    else:
        year_str = year.replace("UL", "20")
    return year_str


def _plot(hist_list, stack_dict,  **kwargs):
    if kwargs.get("debug", False):
        print(f'\t in plot ... kwargs = {kwargs}')

    size = 7
    fig = plt.figure()   # figsize=(size,size/_phi))

    fig.add_axes((0.1, 0.15, 0.85, 0.8))

    _draw_plot(hist_list, stack_dict, **kwargs)

    year_str = get_year_str(year = kwargs.get('year',"RunII"))

    ax = fig.gca()
    hep.cms.label("Internal", data=True,
                  year=year_str, loc=0, ax=ax)


    return fig, ax


def _plot_from_dict(plot_data, **kwargs):
    if kwargs.get("debug", False):
        print(f'\t in plot ... kwargs = {kwargs}')

    size = 7
    fig = plt.figure()   # figsize=(size,size/_phi))

    do_ratio = len(plot_data["ratio"])
    if do_ratio:
        grid = fig.add_gridspec(2, 1, hspace=0.06, height_ratios=[3, 1],
                                left=0.1, right=0.95, top=0.95, bottom=0.1)
        main_ax = fig.add_subplot(grid[0])
    else:
        fig.add_axes((0.1, 0.15, 0.85, 0.8))
        main_ax = fig.gca()
        ratio_ax = None

    year_str = get_year_str(year = kwargs.get('year',"RunII"))

    hep.cms.label("Internal", data=True,
                  year=year_str, loc=0, ax=main_ax)


    _draw_plot_from_dict(plot_data, **kwargs)

    if do_ratio:

        top_xlabel = plt.gca().get_xlabel()
        plt.xlabel("")

        ratio_ax = fig.add_subplot(grid[1], sharex=main_ax)
        plt.setp(main_ax.get_xticklabels(), visible=False)

        central_value_artist = ratio_ax.axhline(
            kwargs.get("ratio_line_value", 1.0),
            color="black",
            linestyle="dashed",
            linewidth=1.0
        )

        for ratio_name, ratio_data in plot_data["ratio"].items():

            print("plotting ratio", ratio_name)

            error_bar_type = ratio_data.get("type", "bar")
            if error_bar_type == "band":

                # Only works for contant bin size !!!!
                bin_width = (ratio_data["centers"][1] - ratio_data["centers"][0])

                # add check for variable bin width!!! TODO

                # Create hatched error regions using fill_between
                for xi, yi, err in zip(ratio_data["centers"], ratio_data["ratio"], ratio_data["error"]):
                    plt.fill_between([xi - bin_width/2, xi + bin_width/2], yi - err, yi + err,
                                     hatch=ratio_data.get("hatch", "/"),
                                     edgecolor=ratio_data.get("color", "black"),
                                     facecolor='none',
                                     linewidth=0.0,
                                     zorder=1)

            else:
                ratio_ax.errorbar(
                    ratio_data["centers"],       # x-values
                    ratio_data["ratio"],       # y-values
                    yerr=ratio_data["error"],
                    color=ratio_data.get("color", "black"),
                    marker=ratio_data.get("marker", "o"),
                    linestyle=ratio_data.get("linestyle", "none"),
                    markersize=ratio_data.get("markersize", 4),
                )

        #
        #  labels / limits
        #
        plt.ylabel(kwargs.get("rlabel", "Ratio"))
        plt.ylabel(plt.gca().get_ylabel(), loc='center')

        plt.xlabel(kwargs.get("xlabel", top_xlabel), loc='right')

        plt.ylim(*kwargs.get('rlim', [0, 2]))

    return fig, main_ax, ratio_ax



def _plot2d(hist, plotConfig, **kwargs):

    if kwargs.get("debug", False):
        print(f'\t in plot ... kwargs = {kwargs}')

    if kwargs.get("full", False):
        fig = plt.figure()   # figsize=(size,size/_phi))
        # fig.add_axes((0.1, 0.15, 0.85, 0.8))
        hist.plot2d_full(
            main_cmap="cividis",
            top_ls="--",
            top_color="orange",
            top_lw=2,
            side_ls=":",
            side_lw=2,
            side_color="steelblue",
        )
    else:
        fig = plt.figure()   # figsize=(size,size/_phi))
        fig.add_axes((0.1, 0.15, 0.85, 0.8))
        hist.plot2d()

    ax = fig.gca()

    hep.cms.label("Internal", data=True,
                  year=kwargs.get('year',"RunII").replace("UL", "20"), loc=0, ax=ax)

    return fig, ax


def _plot_ratio(hist_list, stack_dict, ratio_list, **kwargs):
    r"""
    Takes options:
        "norm"              : bool (False),
        "ratio_line_value"  : number (1.0),
        "uncertainty_type"  : "poisson", "poisson-ratio",
                              "efficiency" ("poisson")
        "rlim"              : [rmin, rmax] ([0,2])
    }
    """

    size = 7
    fig = plt.figure()
    grid = fig.add_gridspec(2, 1, hspace=0.06, height_ratios=[3, 1],
                            left=0.1, right=0.95, top=0.95, bottom=0.1)

    year_str = get_year_str(year = kwargs.get('year',"RunII"))

    main_ax = fig.add_subplot(grid[0])
    hep.cms.label("Internal", data=True,
                  year=year_str, loc=0, ax=main_ax)

    _draw_plot(hist_list, stack_dict, **kwargs)

    top_xlabel = plt.gca().get_xlabel()
    plt.xlabel("")

    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)

    central_value_artist = subplot_ax.axhline(
        kwargs.get("ratio_line_value", 1.0),
        color="black",
        linestyle="dashed",
        linewidth=1.0
    )

    for ir, ratio in enumerate(ratio_list):

        error_bar_type = ratio[3].get("type", "bar")
        if error_bar_type == "band":

            # Only works for contant bin size !!!!
            bin_width = (ratio[0][1] - ratio[0][0])

            # add check for variable bin width!!! TODO

            # Create hatched error regions using fill_between
            for xi, yi, err in zip(ratio[0], ratio[1], ratio[2]):
                plt.fill_between([xi - bin_width/2, xi + bin_width/2], yi - err, yi + err,
                                 hatch=ratio[3].get("hatch", "/"),
                                 edgecolor=ratio[3].get("color", "black"),
                                 facecolor='none',
                                 linewidth=0.0,
                                 zorder=1)

        else:
            subplot_ax.errorbar(
                ratio[0],       # x-values
                ratio[1],       # y-values
                yerr=ratio[2],
                color=ratio[3].get("color", "black"),
                marker=ratio[3].get("marker", "o"),
                linestyle=ratio[3].get("linestyle", "none"),
                markersize=ratio[3].get("markersize", 4),
            )

    #
    #  labels / limits
    #
    plt.ylabel(kwargs.get("rlabel", "Ratio"))
    plt.ylabel(plt.gca().get_ylabel(), loc='center')

    plt.xlabel(kwargs.get("xlabel", top_xlabel), loc='right')

    plt.ylim(*kwargs.get('rlim', [0, 2]))

    return fig, main_ax, subplot_ax




def _makeHistsFromList(cfg, var, cut, region, process, **kwargs):

    if kwargs.get("debug", False):
        print(f" in _makeHistFromList hist process={process}, "
              f"cut={cut}")

    rebin = kwargs.get("rebin", 1)
    var_over_ride = kwargs.get("var_over_ride", {})
    label_override = kwargs.get("labels", None)
    year = kwargs.get("year", "RunII")

    #
    #  Unstacked hists
    #
    hists = []

    #
    #  Parse the Lists
    #
    if type(process) is list:
        process_config = [get_value_nested_dict(cfg.plotConfig, p) for p in process]
    else:
        try:
            process_config = get_value_nested_dict(cfg.plotConfig, process)
        except ValueError:
            raise ValueError(f"\t ERROR process = {process} not in plotConfig! \n")

        var_to_plot = var_over_ride.get(process, var)

    #
    #  cut list
    #
    if type(cut) is list:
        for ic, _cut in enumerate(cut):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), _cut, region)

            _process_config = copy.deepcopy(process_config)
            _process_config["hist_name"] = _process_config["process"] + _cut
            _process_config["fillcolor"] = _colors[ic]
            _process_config["label"]     = get_label(f"{process_config['label']} { _cut}", label_override, ic)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            _hist = get_hist(cfg, _process_config,
                             var=var_to_plot, region=region, cut=_cut, rebin=rebin, year=year,
                             debug=kwargs.get("debug", False))

            hists.append( (_hist, _process_config) )

    #
    #  region list
    #
    elif type(region) is list:
        for ir, _reg in enumerate(region):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), cut, _reg)

            _process_config = copy.deepcopy(process_config)
            _process_config["hist_name"]   = _process_config["process"] + _reg
            _process_config["fillcolor"] = _colors[ir]
            _process_config["label"]     = get_label(f"{process_config['label']} { _reg}", label_override, ir)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            _hist = get_hist(cfg, _process_config,
                             var=var_to_plot, region=_reg, cut=cut, rebin=rebin, year=year,
                             debug=kwargs.get("debug", False))

            hists.append( (_hist, _process_config) )

    #
    #  input file list
    #
    elif len(cfg.hists) > 1 and not cfg.combine_input_files:
        if kwargs.get("debug", False):
            print_list_debug_info(process, process_config.get("tag"), cut, region)

        fileLabels = kwargs.get("fileLabels", [])

        for iF, _input_File in enumerate(cfg.hists):

            _process_config = copy.deepcopy(process_config)
            _process_config["hist_name"] = _process_config["process"] + iF
            _process_config["fillcolor"] = _colors[iF]

            if label_override:
                _process_config["label"] = label_override[iF]
            elif iF < len(fileLabels):
                _process_config["label"] = _process_config["label"] + " " + fileLabels[iF]
            else:
                _process_config["label"] = _process_config["label"] + " file" + str(iF + 1)

            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            _hist = get_hist(cfg, _process_config,
                             var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                             file_index=iF,
                             debug=kwargs.get("debug", False))

            hists.append( (_hist, _process_config) )

    #
    #  process list
    #
    elif type(process) is list:
        for _, _proc_conf in enumerate(process_config):

            if kwargs.get("debug", False):
                print_list_debug_info(_proc_conf["process"], _proc_conf.get("tag"), cut, region)

            _process_config = copy.deepcopy(_proc_conf)
            _process_config["hist_name"]   = _process_config["process"]
            _process_config["fillcolor"] = _proc_conf.get("fillcolor", None)#.replace("yellow", "orange")
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            var_to_plot = var_over_ride.get(_proc_conf["process"], var)

            _hist = get_hist(cfg, _process_config,
                             var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                             debug=kwargs.get("debug", False))

            hists.append( (_hist, _process_config) )


    #
    #  var list
    #
    elif type(var) is list:
        for iv, _var in enumerate(var):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), cut, region)

            _process_config = copy.deepcopy(process_config)
            _process_config["hist_name"] = _process_config["process"] + _var
            _process_config["fillcolor"] = _colors[iv]
            _process_config["label"]     = get_label(f"{process_config['label']} { _var}", label_override, iv)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            _hist = get_hist(cfg, _process_config,
                             var=_var, region=region, cut=cut, rebin=rebin, year=year,
                             debug=kwargs.get("debug", False))

            hists.append( (_hist, _process_config) )

    #
    #  year list
    #
    elif type(year) is list:
        for iy, _year in enumerate(year):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), cut, region)

            _process_config = copy.copy(process_config)
            _process_config["hist_name"] = _process_config["process"] + _year
            _process_config["fillcolor"] = _colors[iy]
            _process_config["label"]     = get_label(f"{process_config['label']} { _year}", label_override, iy)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            _hist = get_hist(cfg, _process_config,
                             var=var, region=region, cut=cut, rebin=rebin, year=_year,
                             debug=kwargs.get("debug", False))

            hists.append( (_hist, _process_config) )


    else:
        raise Exception("Error something needs to be a list!")

    #
    # Create Dict
    #
    plot_data = {} # defaultdict(dict)
    plot_data["hists"] = {}
    plot_data["stack"] = {}
    plot_data["ratio"] = {}
    plot_data["var"] = var
    plot_data["cut"] = cut
    plot_data["region"] = region

    for _h in hists:
        _h_data   = _h[0]
        _h_config_master = _h[1]
        _h_config = copy.deepcopy(_h_config_master)
        _h_config["values"], _h_config["variance"], _h_config["edges"], _h_config["centers"], _h_config["x_label"] = get_hist_data( _h_data )
        plot_data["hists"][_h_config['hist_name']] = _h_config


    if kwargs.get("doRatio", kwargs.get("doratio", False)):

        ratio_plots = []

        denValues = hists[-1][0].values()
        denVars   = hists[-1][0].variances()

        denValues[denValues == 0] = _epsilon
        denCenters = hists[-1][0].axes[0].centers

        # Bkg error band

        band_ratios = np.ones(len(denCenters))
        band_uncert  = np.sqrt(denVars * np.power(denValues, -2.0))
        band_config = {"color": "k",  "type": "band", "hatch": "\\\\",
                       "ratio":band_ratios.tolist(),
                       "error":band_uncert.tolist(),
                       "centers": denCenters.tolist()}
        plot_data["ratio"]["bkg_band"] = band_config

        for iH in range(len(hists) - 1):

            numValues = hists[iH][0].values()
            numVars   = hists[iH][0].variances()

            ratio_config = {"color": _colors[iH],
                            "marker": "o",
                            }
            ratios, ratio_uncert = makeRatio(numValues, numVars, denValues, denVars, **kwargs)
            ratio_config["ratio"] = ratios.tolist()
            ratio_config["error"] = ratio_uncert.tolist()
            ratio_config["centers"] = denCenters.tolist()
            plot_data["ratio"][iH] = ratio_config



    return plot_data


def make_plot_from_dict(plot_data, **kwargs):

    fig, main_ax, ratio_ax = _plot_from_dict(plot_data, **kwargs)
    main_ax.set_title(f"{plot_data['region']}")
    ax = (main_ax, ratio_ax)


    if kwargs.get("outputFolder", None):

        if type(process) is list:
            tagName = "_vs_".join(process)
        else:
            tagName = process_config.get("tag", "fourTag")

        if kwargs.get("yscale", None) == "log":
            _savefig(fig, plot_data["var"]+"_logy", kwargs.get("outputFolder"), kwargs.get("year","RunII"), plot_data["cut"], tagName, plot_data["region"], process)
        else:
            _savefig(fig, plot_data["var"], kwargs.get("outputFolder"), kwargs.get("year","RunII"), plot_data["cut"], tagName, plot_data["region"], process)

    # Specify the file name
    file_name = "data.yaml"

    # Write data to a YAML file
    with open(file_name, "w") as yfile:  # Use "w" for writing in text mode
        yaml.dump(plot_data, yfile)

    return fig, ax


def load_stack_config(stack_config, var, cut, region, **kwargs):

    stack_dict = {}
    var_over_ride = kwargs.get("var_over_ride", {})
    rebin   = kwargs.get("rebin", 1)
    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)

    #
    #  Loop processes in the stack config
    #
    for _proc_name, _proc_config in stack_config.items():

        var_to_plot = var_over_ride.get(_proc_name, var)

        if kwargs.get("debug", False):
            print(f"stack_process is {_proc_name} var is {var_to_plot}")

        #
        #  If this component is a process in the hist_obj
        #
        if _proc_config.get("process", None):


            #
            #  Get the hist object from the input data file(s)
            #
            _hist = get_hist(cfg, _proc_config,
                             var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                             debug=kwargs.get("debug", False))

            stack_dict[_proc_name] = (_hist, _proc_config)

        #
        #  If this compoent is a sum of processes in the hist_obj
        #
        elif _proc_config.get("sum", None):

            hist_sum = None
            for sum_proc_name, sum_proc_config in _proc_config.get("sum").items():

                sum_proc_config["year"] = _proc_config["year"]

                var_to_plot = var_over_ride.get(sum_proc_name, var)

                #
                #  Get the hist object from the input data file(s)
                #
                _hist = get_hist(cfg, sum_proc_config,
                                 var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                                 debug=kwargs.get("debug", False))

                if hist_sum:
                    hist_sum += _hist
                else:
                    hist_sum = _hist

            stack_dict[_proc_name] = (hist_sum, _proc_config)

        else:
            raise Exception("Error need to config either process or sum")

    return stack_dict


def get_ratio_plots(ratio_config, hists, stack_dict, **kwargs):
    ratio_plots = []

    for _, ratio_config in ratio_config.items():

        num_config = ratio_config.get("numerator")
        den_config = ratio_config.get("denominator")
        numValues, numVars, numCenters = get_values_variances_centers_from_dict(num_config, hists, stack_dict)
        denValues, denVars, _          = get_values_variances_centers_from_dict(den_config, hists, stack_dict)

        if kwargs.get("norm", False):
            ratio_config["norm"] = True

        #
        # Clean den
        #
        ratios, ratio_uncert = makeRatio(numValues, numVars, denValues, denVars, **ratio_config)
        ratio_plots.append((numCenters, ratios, ratio_uncert, ratio_config))

        # Bkg error band
        default_band_config = {"color": "k",  "type": "band", "hatch": "\\\\\\"}
        band_config = ratio_config.get("bkg_err_band", default_band_config)

        if band_config:
            band_ratios = np.ones(len(numCenters))
            band_uncert  = np.sqrt(denVars * np.power(denValues, -2.0))
            ratio_plots.append((numCenters, band_ratios, band_uncert, band_config))



    return ratio_plots


def makePlot(cfg, var='selJets.pt',
             cut="passPreSel", region="SR", **kwargs):
    r"""
    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       cut      : "passPreSel",
       region   : "SR",

       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    process = kwargs.get("process", None)
    rebin   = kwargs.get("rebin", 1)
    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)
    if debug: print(f"In makePlot kwargs={kwargs}")

    if (type(cut) is list) or (type(region) is list) or (len(cfg.hists) > 1 and not cfg.combine_input_files) or (type(var) is list) or (type(process) is list) or (type(year) is list):
        try:
            plot_data =  _makeHistsFromList(cfg, var, cut, region, **kwargs)
            return make_plot_from_dict(plot_data, **kwargs)

        except ValueError as e:
            raise ValueError(e)

    # Make process a list if it exits and isnt one already
    if process is not None and type(process) is not list:
        process = [process]

    #
    #  Lets you plot different variables for differnet processes
    #
    var_over_ride = kwargs.get("var_over_ride", {})

    if cut not in cfg.cutList:
        raise AttributeError(f"{cut} not in cutList {cfg.cutList}")

    tagNames = []

    #
    #  Unstacked hists
    #
    hists = []
    hist_config = cfg.plotConfig["hists"]
    if process is not None:
        hist_config = {key: hist_config[key] for key in process if key in hist_config}

    #
    #  Loop of hists in config file
    #
    for _proc_name, _proc_config in hist_config.items():

        #
        #  Add name to config
        #
        _proc_config["name"] = _proc_name

        #
        #  Used for naming output pdf
        #
        tagNames.append(_proc_config["tag"])

        var_to_plot = var_over_ride.get(_proc_name, var)

        #
        #  Get the hist object from the input data file(s)
        #
        _hist = get_hist(cfg, _proc_config,
                         var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                         debug=kwargs.get("debug", False))

        hists.append( (_hist, _proc_config) )


    #
    #  The stack
    #
    stack_config = cfg.plotConfig.get("stack", {})
    if process is not None:
        stack_config = {key: stack_config[key] for key in process if key in stack_config}


    stack_dict = load_stack_config(stack_config, var, cut, region, **kwargs)

    if len(tagNames) == 0:
        tagNames.append( get_value_nested_dict(stack_config,"tag") )

    #
    #  Config Ratios
    #
    if kwargs.get("doRatio", kwargs.get("doratio", False)):
        ratio_config = cfg.plotConfig["ratios"]
        ratio_plots = get_ratio_plots(ratio_config, hists, stack_dict, **kwargs)

        fig, main_ax, ratio_ax = _plot_ratio(hists, stack_dict, ratio_plots,  **kwargs)
        main_ax.set_title(f"{region}")
        ax = (main_ax, ratio_ax)
    else:
        fig, ax = _plot(hists, stack_dict, **kwargs)
        ax.set_title(f"{region}")

    #
    # Save Fig
    #
    if kwargs.get("outputFolder", None):
        tagName = "fourTag" if "fourTag" in tagNames else "threeTag"
        if kwargs.get("yscale", "linear") == "log":
            _savefig(fig, var+"_logy", kwargs.get("outputFolder"), kwargs.get("year","RunII"), cut, tagName, region)
        else:
            _savefig(fig, var, kwargs.get("outputFolder"), kwargs.get("year","RunII"), cut, tagName, region)

    return fig, ax


def make2DPlot(cfg, process, var='selJets.pt',
               cut="passPreSel", region="SR", **kwargs):
    r"""
    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       year     : "2017",
       cut      : "passPreSel",
       region   : "SR",

       plotting opts
        'rebin'    : int (1),
    """

    if len(cfg.hists) > 1:
        #
        # Find which file has the process we are looking for
        #
        process_config = get_value_nested_dict(cfg.plotConfig, process)
        process_name = process_config["process"]
        for _input_data in cfg.hists:
            _hist_to_plot = _input_data['hists'][var]
            if process_name in _hist_to_plot.axes["process"]:
                hist_to_plot = _hist_to_plot

    else:
        input_data = cfg.hists[0]
        hist_to_plot = input_data['hists'][var]

    cut_dict = get_cut_dict(cut, cfg.cutList)

    #
    #  Get the year
    #    (Got to be a better way to do this....)
    #
    year = kwargs.get("year","RunII")
    year = sum if year == "RunII" else year

    #
    #  Unstacked hists
    #
    process_config = get_value_nested_dict(cfg.plotConfig, process)
    tagName = process_config.get("tag", "fourTag")
    tag = cfg.plotConfig["codes"]["tag"][tagName]
    # labels.append(v.get("label"))
    # hist_types. append(v.get("histtype", "errorbar"))

    # region_selection = sum if region in ["sum", sum] else hist.loc(codes["region"][region])

    if region in ["sum", sum]:
        region_selection = sum
    elif type(cfg.plotConfig["codes"]["region"][region]) is list:
        region_selection = [hist.loc(_r) for _r in cfg.plotConfig["codes"]["region"][region]]
    else:
        region_selection = hist.loc(cfg.plotConfig["codes"]["region"][region])

    if kwargs.get("debug", False):
        print(f" hist process={process}, "
              f"tag={tag}, year={year}")

    varName = hist_to_plot.axes[-1].name
    hist_dict = {"process": process_config["process"],
                 "year":    year,
                 "tag":     hist.loc(tag),
                 "region":  region_selection,
                 varName:   hist.rebin(kwargs.get("rebin", 1))}

    hist_dict = hist_dict | cut_dict
    _hist = hist_to_plot[hist_dict]

    if len(_hist.shape) == 3:  # for 2D plots
        _hist = _hist[sum, :, :]


    #
    # Make the plot
    #
    fig, ax = _plot2d(_hist, cfg.plotConfig, **kwargs)
    ax.set_title(f"{region}  ({cut})", fontsize=16)

    #
    # Save Fig
    #
    if kwargs.get("outputFolder", None):
        _savefig(fig, var, kwargs.get("outputFolder"),
                 kwargs.get("year","RunII"), cut, tagName, region, process)

    return fig, ax


def init_arg_parser():

    parser = argparse.ArgumentParser(description='plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(dest="inputFile",
                        default='hists.pkl', nargs='+',
                        help='Input File. Default: hists.pkl')

    parser.add_argument('-l', '--labelNames', dest="fileLabels",
                        default=["fileA", "fileB"], nargs='+',
                        help='label Names when more than one input file')

    parser.add_argument('-o', '--outputFolder', default=None,
                        help='Folder for output folder. Default: plots/')

    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="analysis/metadata/plotsAll.yml",
                        help='Metadata file.')

    parser.add_argument('--modifiers', dest="modifiers",
                        default="analysis/metadata/plotModifiers.yml",
                        help='Metadata file.')

    parser.add_argument('-s', '--skip', dest="skip_hists",
                        default=[], nargs='+',
                        help='Name of hists to skip')

    parser.add_argument('--doTest', action="store_true", help='Metadata file.')
    parser.add_argument('--debug', action="store_true", help='')
    parser.add_argument('--signal', action="store_true", help='')
    parser.add_argument('--combine_input_files', action="store_true", help='')


    return parser

def parse_args():

    parser = init_arg_parser()

    args = parser.parse_args()
    return args


def load_hists(input_hists):
    hists = []
    for _inFile in input_hists:
        with open(_inFile, 'rb') as hfile:
            hists.append(load(hfile))

    return hists


def read_axes_and_cuts(hists, plotConfig):

    axisLabels = {}
    cutList = []

    axisLabels["var"] = hists[0]['hists'].keys()
    var1 = list(hists[0]['hists'].keys())[0]

    for a in hists[0]["hists"][var1].axes:
        axisName = a.name
        if axisName == var1:
            continue

        if isinstance(a, hist.axis.Boolean):
            cutList.append(axisName)
            continue

        if a.extent > 20:
            continue   # HACK to skip the variable bins FIX

        axisLabels[axisName] = []

        for iBin in range(a.extent):

            if axisName in plotConfig["codes"]:
                if a.value(iBin) not in plotConfig["codes"][axisName]:
                    continue
                value = plotConfig["codes"][axisName][a.value(iBin)]
            else:
                value = a.value(iBin)

            axisLabels[axisName].append(value)

    return axisLabels, cutList


def print_cfg(cfg):
    print("Regions...")
    for reg in cfg.plotConfig["codes"]["region"].keys():
        if type(reg) is str:
            print(f"\t{reg}")

    print("Cuts...")
    for c in cfg.cutList:
        print(f"\t{c}")

    print("Processes...")
    for key, values in cfg.plotConfig.items():
        if key in ["hists", "stack"]:
            for _key, _ in values.items():
                print(f"\t{_key}")
