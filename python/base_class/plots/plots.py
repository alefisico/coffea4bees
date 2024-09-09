import os
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

_phi = (1 + np.sqrt(5)) / 2
_epsilon = 0.001
_colors = ["xkcd:blue", "xkcd:red", "xkcd:off green",
           "xkcd:orange", "xkcd:violet", "xkcd:grey",
           "xkcd:black", "xkcd:pink" , "xkcd:pale blue"]


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


def get_value_nested_dict(nested_dict, target_key, default=None):
    """ Return the first value from mathching key from nested dict
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v

        if type(v) is dict:
            return_value = get_value_nested_dict(v, target_key, default)
            if not return_value == default:
                return return_value

    return default


def get_values_centers_from_dict(hist_config, hists, stack_dict):

    if hist_config["type"] == "hists":

        for h_data, h_config in hists:

            if h_config["name"] == hist_config["key"]:
                return h_data.values(), h_data.axes[0].centers

        print(f"ERROR: input to ratio of key {hist_config['key']} not found in hists")

    if hist_config["type"] == "stack":
        stack_dict_for_hist = {k: v[0] for k, v in stack_dict.items() }
        hStackHists = list(stack_dict_for_hist.values())
        return_values = [h.values() for h in hStackHists]
        return_values = np.sum(return_values, axis=0)
        return return_values, hStackHists[0].axes[0].centers

    print("ERROR: ratio needs to be of type 'hists' or 'stack'")


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

    if type(codes["region"][region]) is list:
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
        print(f"ERROR did not find var {var} with process {config['process']} in inputs")

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


def makeRatio(numValues, denValues, **kwargs):

    denValues[denValues == 0] = _epsilon
    ratios = numValues / denValues

    if kwargs.get("norm", False):
        numSF = np.sum(numValues, axis=0)
        denSF = np.sum(denValues, axis=0)
        ratios *= denSF / numSF

    # Set 0 and inf to nan to hide during plotting
    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan

    # https://github.com/scikit-hep/hist/blob/main/src/hist/intervals.py
    ratio_uncert = ratio_uncertainty(
        num=numValues,
        denom=denValues,
        uncertainty_type=kwargs.get("uncertainty_type", "poisson"),
    )

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


def _plot(hist_list, stack_dict, plotConfig, **kwargs):
    if kwargs.get("debug", False):
        print(f'\t in plot ... kwargs = {kwargs}')

    size = 7
    fig = plt.figure()   # figsize=(size,size/_phi))

    fig.add_axes((0.1, 0.15, 0.85, 0.8))

    _draw_plot(hist_list, stack_dict, **kwargs)

    ax = fig.gca()
    hep.cms.label("Internal", data=True,
                  year=kwargs.get('year',"RunII").replace("UL", "20"), loc=0, ax=ax)


    return fig, ax


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

    main_ax = fig.add_subplot(grid[0])
    hep.cms.label("Internal", data=True,
                  year=kwargs.get('year',"RunII").replace("UL", "20"), loc=0, ax=main_ax)

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
        process_config = get_value_nested_dict(cfg.plotConfig, process)
        var_to_plot = var_over_ride.get(process, var)

    #
    #  cut list
    #
    if type(cut) is list:
        for ic, _cut in enumerate(cut):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), _cut, region)

            _process_config = copy.copy(process_config)
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

            _process_config = copy.copy(process_config)
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
    elif len(cfg.hists) > 1:
        if kwargs.get("debug", False):
            print_list_debug_info(process, process_config.get("tag"), cut, region)

        fileLabels = kwargs.get("fileLabels", [])

        for iF, _input_File in enumerate(cfg.hists):

            _process_config = copy.copy(process_config)
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

            _process_config = copy.copy(_proc_conf)
            _process_config["fillcolor"] = _proc_conf.get("fillcolor", None).replace("yellow", "orange")
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

            _process_config = copy.copy(process_config)
            _process_config["fillcolor"] = _colors[iv]
            _process_config["label"]     = get_label(f"{process_config['label']} { _var}", label_override, iv)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            _hist = get_hist(cfg, _process_config,
                             var=_var, region=region, cut=cut, rebin=rebin, year=year,
                             debug=kwargs.get("debug", False))

            hists.append( (_hist, _process_config) )

    else:
        raise Exception("Error something needs to be a list!")


    if kwargs.get("doRatio", kwargs.get("doratio", False)):

        ratio_plots = []

        denValues = hists[-1][0].values()

        denValues[denValues == 0] = _epsilon
        denCenters = hists[-1][0].axes[0].centers

        for iH in range(len(hists) - 1):

            numValues = hists[iH][0].values()

            ratio_config = {"color": _colors[iH],
                            "marker": "o",
                            }
            ratios, ratio_uncert = makeRatio(numValues, denValues, **kwargs)
            ratio_plots.append((denCenters, ratios, ratio_uncert, ratio_config))

        fig, main_ax, ratio_ax = _plot_ratio(hists, {}, ratio_plots, **kwargs)
        main_ax.set_title(f"{region}")
        ax = (main_ax, ratio_ax)
    else:
        fig, ax = _plot(hists, {}, cfg.plotConfig, **kwargs)
        ax.set_title(f"{region}")

    if kwargs.get("outputFolder", None):

        if type(process) is list:
            tagName = "_vs_".join(process)
        else:
            tagName = process_config.get("tag", "fourTag")

        if kwargs.get("yscale", None) == "log":
            _savefig(fig, var+"_logy", kwargs.get("outputFolder"), kwargs.get("year","RunII"), cut, tagName, region, process)
        else:
            _savefig(fig, var, kwargs.get("outputFolder"), kwargs.get("year","RunII"), cut, tagName, region, process)

    return fig, ax


def makePlot(cfg, var='selJets.pt',
             cut="passPreSel", region="SR", year="RunII", **kwargs):
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


    if (type(cut) is list) or (type(region) is list) or (len(cfg.hists) > 1 and not cfg.combine_input_files) or (type(var) is list) or (type(process) is list):
        return _makeHistsFromList(cfg, var, cut, region, **kwargs)

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
    # Add args
    #

    #yearName = get_value_nested_dict(cfg.plotConfig,  "year", default="RunII")
    #

    #
    #  The stack
    #
    stack_dict = {}
    stack_config = cfg.plotConfig.get("stack", {})
    if process is not None:
        stack_config = {key: stack_config[key] for key in process if key in stack_config}

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

            tagNames.append(_proc_config["tag"])

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

    #
    #  Config Ratios
    #
    if kwargs.get("doRatio", kwargs.get("doratio", False)):
        ratio_config = cfg.plotConfig["ratios"]
        ratio_plots = []

        for _, ratio_config in ratio_config.items():

            num_config = ratio_config.get("numerator")
            den_config = ratio_config.get("denominator")

            numValues, numCenters = get_values_centers_from_dict(num_config, hists, stack_dict)
            denValues, _          = get_values_centers_from_dict(den_config, hists, stack_dict)

            if kwargs.get("norm", False):
                ratio_config["norm"] = True

            #
            # Clean den
            #
            ratios, ratio_uncert = makeRatio(numValues, denValues, **ratio_config)

            ratio_plots.append((numCenters, ratios, ratio_uncert, ratio_config))

        fig, main_ax, ratio_ax = _plot_ratio(hists, stack_dict, ratio_plots,  **kwargs)
        main_ax.set_title(f"{region}")
        ax = (main_ax, ratio_ax)
    else:
        fig, ax = _plot(hists, stack_dict, cfg.plotConfig, **kwargs)
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
        input_data = cfg.hists
    else:
        input_data = cfg.hists[0]

    hist_to_plot = input_data['hists'][var]
    varName = hist_to_plot.axes[-1].name
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
    ax.set_title(f"{region}")

    #
    # Save Fig
    #
    if kwargs.get("outputFolder", None):
        _savefig(fig, var, kwargs.get("outputFolder"),
                 kwargs.get("year","RunII"), cut, tagName, region, process)

    return fig, ax


def parse_args():

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
