import os
import hist
import yaml
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
           "xkcd:orange", "xkcd:violet", "xkcd:grey"]


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


def get_values_centers_from_dict(input_dict, hist_index, hists, stack_dict):
    if input_dict["type"] == "hists":
        return hists[hist_index[input_dict["key"]]].values(), hists[hist_index[input_dict["key"]]].axes[0].centers

    if input_dict["type"] == "stack":
        hStackHists = list(stack_dict.values())
        return_values = [h.values() for h in hStackHists]
        return_values = np.sum(return_values, axis=0)
        return return_values, hStackHists[0].axes[0].centers

    print("ERROR: ratio needs to be of type 'hists' or 'stack'")


def _savefig(fig, var, *args):
    outputPath = "/".join(args)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    varStr = var if type(var) == str else "_vs_".join(var)
    fig.savefig(outputPath + "/" + varStr.replace(".", '_') + ".pdf")
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


def get_hist(input_data, var, process):

    if type(input_data) is list:
        for _input_data in input_data:
            if process in _input_data['hists'][var].axes["process"]:
                return _input_data['hists'][var]

    return input_data['hists'][var]





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
    if len(stack_dict):
        s = hist.Stack.from_dict(stack_dict)

        s.plot(stack=True, histtype="fill",
               color=kwargs.get("stack_colors_fill"),
               label=None,
               density=norm)

        s.plot(stack=True, histtype="step",
               color=kwargs.get("stack_colors_edge"),
               label=None,
               density=norm)

    stack_patches = []
    for ik, k in enumerate(kwargs.get("stack_labels")):
        if k in ["None"]:
            continue
        stack_patches.append(mpatches.Patch(facecolor=kwargs.get("stack_colors_fill")[ik],
                                            edgecolor=kwargs.get("stack_colors_edge")[ik],
                                            label=k))

    #
    #  Draw the hists
    #
    hist_labels = kwargs.get("hist_labels")
    hist_fill_colors = kwargs.get("hist_fill_colors")
    hist_types = kwargs.get("hist_types")
    hist_artists = []

    for ih, h in enumerate(hist_list):
        this_plot_options = {"density": norm,
                             "label": hist_labels[ih],
                             "color": hist_fill_colors[ih],
                             "histtype": hist_types[ih],
                             "yerr": False,
                             }
        if hist_types[ih] in ["errorbar"]:
            this_plot_options["markersize"] = 12
            this_plot_options["yerr"] = True

        hist_artists.append(h.plot(**this_plot_options)[0])

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
                  year=kwargs['year'].replace("UL", "20"), loc=0, ax=ax)

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
                  year=kwargs['year'].replace("UL", "20"), loc=0, ax=ax)

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
                  year=kwargs['year'].replace("UL", "20"), loc=0, ax=main_ax)

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
            color=kwargs.get("ratio_colors")[ir],
            marker=kwargs.get("ratio_markers")[ir],
            linestyle="none",
            markersize=4,
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
        print(f" hist process={process}, "
              f"cut={cut}")

    if len(cfg.hists) > 1:
        input_hist_File = cfg.hists
    else:
        input_hist_File = cfg.hists[0]

    rebin = kwargs.get("rebin", 1)
    plotConfig = cfg.plotConfig
    codes = cfg.plotConfig["codes"]

    #
    #  Get the year
    #    (Got to be a better way to do this....)
    #
    yearStr = get_value_nested_dict(plotConfig, "year", default="RunII")
    year = sum if yearStr == "RunII" else yearStr
    year_dict = {"year": year}

    #
    #  Unstacked hists
    #
    hists = []
    hist_colors_fill = []
    hist_colors_edge = []
    hist_labels = []
    hist_types = []

    #
    #  Parse the Lists
    #
    if type(cut) is list:
        cutName = "_vs_".join(cut)
        cut_dict = None
    else:
        cutName = cut
        cut_dict = get_cut_dict(cut, cfg.cutList)

    if type(process) is list:
        process_config = [get_value_nested_dict(plotConfig, p) for p in process]
        this_tagName = "_vs_".join(process)
        process_dict = None
        tag_dict = None
        label = None
    else:
        process_config = get_value_nested_dict(plotConfig, process)
        this_tagName = process_config.get("tag", "fourTag")
        this_tag = plotConfig["codes"]["tag"][this_tagName]
        tag_dict = {"tag": hist.loc(this_tag)}
        process_dict = {"process": process_config["process"]}
        label = process if process_config.get("label").lower() == "none" else process_config.get("label")

    if type(region) is list:
        region_dict = None
        regionName = "_vs_".join(region)
    else:
        regionName = region

        if type(codes["region"][region]) is list:
            region_dict = {"region": [hist.loc(_r) for _r in codes["region"][region]] }
        else:
            region_dict = {"region": hist.loc(codes["region"][region])}


    if type(var) is list:
        varName  = None
        var_dict = None
    else:
        if type(input_hist_File) is list:
            varName = input_hist_File[0]['hists'][var].axes[-1].name
        else:
            varName  = input_hist_File['hists'][var].axes[-1].name
        var_dict = {varName: hist.rebin(rebin)}


    #
    #  cut list
    #
    if type(cut) is list:
        for ic, _cut in enumerate(cut):

            if kwargs.get("debug", False):
                print_list_debug_info(process, this_tag, _cut, region)

            hist_colors_fill.append(_colors[ic])
            hist_labels.append(label + " " + _cut)
            hist_types. append("errorbar")

            this_cut_dict = get_cut_dict(_cut, cfg.cutList)
            this_hist_dict = process_dict | tag_dict | region_dict | year_dict | var_dict | this_cut_dict

            this_hist = input_hist_File['hists'][var][this_hist_dict]
            if len(this_hist.shape) == 2:
                this_hist = this_hist[sum,:]

            hists.append(this_hist)
            hists[-1] *= process_config.get("scalefactor", 1.0)

    #
    #  region list
    #
    elif type(region) is list:
        for ir, _reg in enumerate(region):
            if kwargs.get("debug", False):
                print_list_debug_info(process, this_tag, cut, _reg)

            hist_colors_fill.append(_colors[ir])
            hist_labels.append(f"{label} {_reg}")
            hist_types. append("errorbar")

            if type(codes["region"][_reg]) is list:
                this_region_dict = {"region": [hist.loc(_r) for _r in codes["region"][_reg]] }
            else:
                this_region_dict = {"region": hist.loc(codes["region"][_reg])} ### check this is a list

            this_hist_dict = process_dict | tag_dict | this_region_dict | year_dict | var_dict | cut_dict

            this_hist = input_hist_File['hists'][var][this_hist_dict]
            if len(this_hist.shape) == 2:
                this_hist = this_hist[sum,:]

            hists.append(this_hist)
            hists[-1] *= process_config.get("scalefactor", 1.0)

    #
    #  input file list
    #
    elif type(input_hist_File) is list:
        if kwargs.get("debug", False):
            print_list_debug_info(process, this_tag, cut, region)

        this_hist_dict = process_dict | tag_dict | region_dict | year_dict | var_dict | cut_dict

        fileLabels = kwargs.get("fileLabels", [])

        for iF, _input_File in enumerate(input_hist_File):

            hist_colors_fill.append(_colors[iF])
            if iF < len(fileLabels):
                hist_labels.append(label + " " + fileLabels[iF])
            else:
                hist_labels.append(label + " file" + str(iF + 1))
            hist_types. append("errorbar")

            this_hist = input_hist_File[iF]['hists'][var][this_hist_dict]
            if len(this_hist.shape) == 2:
                this_hist = this_hist[sum,:]

            hists.append(this_hist)
            hists[-1] *= process_config.get("scalefactor", 1.0)

    #
    #  process list
    #
    elif type(process) is list:
        for _, _proc_conf in enumerate(process_config):
            label = _proc_conf.get("process") if _proc_conf.get("label").lower() == "none" else _proc_conf.get("label")

            _tagName = _proc_conf.get("tag", "fourTag")
            this_tag = plotConfig["codes"]["tag"][_tagName]

            if kwargs.get("debug", False):
                print_list_debug_info(_proc_conf["process"], this_tag, cut, region)

            hist_colors_fill.append(_proc_conf.get("fillcolor", None).replace("yellow", "orange"))
            hist_labels.append(label)
            hist_types. append("errorbar")

            this_process_dict = {"process": _proc_conf["process"]}
            this_tag_dict     = {"tag":     hist.loc(this_tag)}

            this_hist_dict = this_process_dict | this_tag_dict | region_dict | year_dict | var_dict | cut_dict

            this_hist = input_hist_File['hists'][var][this_hist_dict]
            if len(this_hist.shape) == 2:
                this_hist = this_hist[sum,:]

            hists.append(this_hist)

            # hists.append(input_hist_File['hists'][var][this_hist_dict])
            hists[-1] *= _proc_conf.get("scalefactor", 1.0)

    #
    #  var list
    #
    elif type(var) is list:
        for iv, _var in enumerate(var):

            varName = input_hist_File['hists'][_var].axes[-1].name

            if kwargs.get("debug", False):
                print_list_debug_info(process, this_tag, cut, region)

            hist_colors_fill.append(_colors[iv])

            hist_labels.append(label + " " + _var)
            hist_types. append("errorbar")

            this_var_dict = {varName: hist.rebin(rebin)}

            this_hist_dict = process_dict | tag_dict | region_dict | year_dict | this_var_dict | cut_dict

            this_hist = input_hist_File['hists'][_var][this_hist_dict]
            if len(this_hist.shape) == 2:
                this_hist = this_hist[sum,:]

            hists.append(this_hist)
            hists[-1] *= process_config.get("scalefactor", 1.0)

    else:
        raise Exception("Error something needs to be a list!")

    #
    # Add args
    #
    kwargs["year"] = yearStr
    kwargs["hist_fill_colors"] = hist_colors_fill
    kwargs["hist_labels"] = hist_labels
    kwargs["hist_types"] = hist_types
    kwargs["stack_labels"] = []
    if kwargs.get("doRatio", False):

        ratio_plots = []
        ratio_colors = []
        ratio_markers = []

        denValues = hists[-1].values()

        denValues[denValues == 0] = _epsilon
        denCenters = hists[-1].axes[0].centers

        for iH in range(len(hists) - 1):

            numValues = hists[iH].values()

            ratios, ratio_uncert = makeRatio(numValues, denValues, **kwargs)

            ratio_plots.append((denCenters, ratios, ratio_uncert))
            ratio_colors.append(_colors[iH])
            ratio_markers.append("o")

        kwargs["ratio_colors"]  = ratio_colors
        kwargs["ratio_markers"] = ratio_markers

        fig, main_ax, ratio_ax = _plot_ratio(hists, {}, ratio_plots, **kwargs)
        ax = (main_ax, ratio_ax)
    else:
        fig, ax = _plot(hists, {}, plotConfig, **kwargs)

    if kwargs.get("outputFolder", None):
        _savefig(fig, var, kwargs.get("outputFolder"), yearStr, cutName, this_tagName, regionName, process)

    return fig, ax


def makePlot(cfg, var='selJets.pt',
             cut="passPreSel", region="SR", **kwargs):
    r"""
    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       year     : "2017",
       cut      : "passPreSel",
       region   : "SR",

       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """
    process = kwargs.get("process", None)
    rebin = kwargs.get("rebin", 1)

    if len(cfg.hists) > 1:
        input_data = cfg.hists
    else:
        input_data = cfg.hists[0]

    plotConfig = cfg.plotConfig

    if (type(cut) is list) or (type(region) is list) or (type(input_data) is list and not cfg.combine_input_files) or (type(var) is list): # or (type(process) is list) \
        return _makeHistsFromList(cfg, var, cut, region, **kwargs)

    ### Converts process to list and sends thru same broken mechanism???
    if process and type(process) is not list:
        process = [process]
        kwargs["process"] = process

    codes = plotConfig["codes"]

    #
    #  Lets you plot different variables for differnet processes
    #
    var_over_ride = kwargs.get("var_over_ride", {})

    if cut not in cfg.cutList:
        raise AttributeError(f"{cut} not in cutList {cfg.cutList}")

    cut_dict = get_cut_dict(cut, cfg.cutList)

    if type(codes["region"][region]) is list:
        region_dict = {"region":  [hist.loc(r) for r in codes["region"][region]]}
    else:
        region_dict = {"region":  hist.loc(codes["region"][region])}

    tagNames = []

    #
    #  Unstacked hists
    #
    hists = []
    hist_config = plotConfig["hists"]
    if process is not None:
        hist_config = {key: hist_config[key] for key in process if key in hist_config}

    hist_colors_fill = []
    hist_colors_edge = []
    hist_labels = []
    hist_types = []
    hist_index = {}

    #
    #  Loop of hists in config file
    #
    for _proc_name, _proc_config in hist_config.items():

        #
        #  Needed for ratio
        #
        hist_index[_proc_name] = len(hists)

        #
        #  Used for naming output pdf
        #
        tagNames.append(_proc_config["tag"])


        #
        #  Plotting Style
        #
        hist_colors_fill.append(_proc_config.get('fillcolor', 'k'))
        hist_colors_edge.append(_proc_config.get('edgecolor', 'k'))
        hist_labels     .append(_proc_config.get("label", ""))
        hist_types     . append(_proc_config.get("histtype", "errorbar"))

        var_to_plot = var_over_ride.get(_proc_name, var)
        _year = sum if _proc_config["year"] == "RunII" else _proc_config["year"]
        _tag = plotConfig["codes"]["tag"][_proc_config["tag"]]

        if kwargs.get("debug", False):
            print(f" hist process={_proc_config['process']}, "
                  f"tag={_tag}, year={_year}, var={var_to_plot}")


        _hist_config = {"process": _proc_config['process'],
                        "year":  _year,
                        "tag":   hist.loc(_tag),
                        }

        _hist_config = _hist_config | region_dict | cut_dict

        #
        #  Get the hist object from the input data file(s)
        #
        _hist_obj = get_hist(input_data, var_to_plot, _proc_config['process'])

        #
        #  Add rebin Options
        #
        varName = _hist_obj.axes[-1].name
        var_dict = {varName: hist.rebin(rebin)}
        _hist_config = _hist_config | var_dict

        #
        #  Do the hist selection/binngin
        #
        _hist     = _hist_obj[_hist_config]

        #
        # Catch list vs hist
        #  Shape give (nregion, nBins)
        #
        if len(_hist.shape) == 2:
            _hist = _hist[sum,:]

        hists.append(_hist)

        #
        # Apply Scale factor
        #
        hists[-1] *= _proc_config.get("scalefactor", 1.0)

    #
    # Add args
    #
    yearName = get_value_nested_dict(plotConfig,  "year", default="RunII")
    kwargs["year"] = yearName
    kwargs["hist_fill_colors"] = hist_colors_fill
    kwargs["hist_edge_colors"] = hist_colors_edge
    kwargs["hist_labels"] = hist_labels
    kwargs["hist_types"] = hist_types

    #
    #  The stack
    #
    stack_dict = {}
    stack_config = plotConfig.get("stack", {})
    if process is not None:
        stack_config = {key: stack_config[key] for key in process if key in stack_config}
    stack_colors_fill = []
    stack_colors_edge = []
    stack_labels = []

    #
    #  Loop processes in the stack config
    #
    for _proc_name, _proc_config in stack_config.items():

        _year = sum if _proc_config["year"] == "RunII" else _proc_config["year"]
        year_dict = {"year": _year}

        stack_labels     .append(_proc_config.get('label'))
        stack_colors_fill.append(_proc_config.get('fillcolor'))
        stack_colors_edge.append(_proc_config.get('edgecolor'))

        var_to_plot = var_over_ride.get(_proc_name, var)

        if kwargs.get("debug", False): print(f"stack_process is {_proc_name} var is {var_to_plot}")

        #
        #  If this compoent is a process in the hist_obj
        #
        if _proc_config.get("process", None):

            tagNames.append(_proc_config["tag"])

            _tag = plotConfig["codes"]["tag"][_proc_config["tag"]]

            if kwargs.get("debug", False):
                print("Drawing")
                print(f" stack_config process={_proc_config['process']},"
                    f"tag={_tag}, year={_year}")

            _hist_config = {"process": _proc_config['process'],
                            "tag": hist.loc(_tag),
                            }

            _hist_config = _hist_config | region_dict | year_dict | cut_dict

            #
            #  Get the hist object from the input data file(s)
            #
            _hist_obj = get_hist(input_data, var_to_plot, _proc_config['process'])

            #
            #  Add rebin Options
            #
            varName = _hist_obj.axes[-1].name
            var_dict = {varName: hist.rebin(rebin)}
            _hist_config = _hist_config | var_dict

            #
            #  Do the hist selection/binngin
            #
            _hist     = _hist_obj[_hist_config]

            #
            # Catch list vs hist
            #  Shape give (nregion, nBins)
            #
            if len(_hist.shape) == 2:
                _hist = _hist[sum,:]

            stack_dict[_proc_name] = _hist

        #
        #  If this compoent is a sum of processes in the hist_obj
        #
        elif _proc_config.get("sum", None):

            hist_sum = None
            for sum_proc_name, sum_proc_config in _proc_config.get("sum").items():

                tagNames.append(sum_proc_config["tag"])
                _tag = plotConfig["codes"]["tag"][sum_proc_config["tag"]]

                var_to_plot = var_over_ride.get(sum_proc_name, var)

                _hist_opts = {"process": sum_proc_config['process'],
                                  "tag": hist.loc(_tag),
                                }

                _hist_config = _hist_config | region_dict | year_dict | cut_dict

                #
                #  Get the hist object from the input data file(s)
                #
                _hist_obj = get_hist(input_data, var_to_plot, sum_proc_config['process'])

                #
                #  Add rebin Options
                #
                varName = _hist_obj.axes[-1].name
                var_dict = {varName: hist.rebin(rebin)}
                _hist_config = _hist_config | var_dict

                #
                #  Do the hist selection/binngin
                #
                _hist     = _hist_obj[_hist_config]


                #
                # Catch list vs hist
                #  Shape give (nregion, nBins)
                #
                if len(_hist.shape) == 2:
                    _hist = _hist[sum,:]

                _hist *= sum_proc_config.get("scalefactor", 1.0)

                if hist_sum:
                    hist_sum += _hist
                else:
                    hist_sum = _hist

            stack_dict[_proc_name] = hist_sum

        else:
            raise Exception("Error need to config either process or sum")
    #
    # Pass colors
    #
    kwargs["stack_colors_fill"] = stack_colors_fill
    kwargs["stack_colors_edge"] = stack_colors_edge
    kwargs["stack_labels"]      = stack_labels


    #
    #  Config Ratios
    #
    if kwargs.get("doRatio", False):
        ratio_config = plotConfig["ratios"]
        ratio_plots = []
        ratio_colors = []
        ratio_markers = []

        for k, v in ratio_config.items():

            numDict = v.get("numerator")
            denDict = v.get("denominator")

            numValues, numCenters = get_values_centers_from_dict(numDict, hist_index, hists, stack_dict)
            denValues, _          = get_values_centers_from_dict(denDict, hist_index, hists, stack_dict)

            if kwargs.get("norm", False):
                v["norm"] = True

            #
            # Clean den
            #
            ratios, ratio_uncert = makeRatio(numValues, denValues, **v)

            ratio_plots.append((numCenters, ratios, ratio_uncert))
            ratio_colors.append(v.get("color", "black"))
            ratio_markers.append(v.get("marker", "o"))

        kwargs["ratio_colors"]  = ratio_colors
        kwargs["ratio_markers"] = ratio_markers

        fig, main_ax, ratio_ax = _plot_ratio(hists, stack_dict, ratio_plots,  **kwargs)
        ax = (main_ax, ratio_ax)
    else:
        fig, ax = _plot(hists, stack_dict, plotConfig, **kwargs)

    #
    # Save Fig
    #
    if kwargs.get("outputFolder", None):
        tagName = "fourTag" if "fourTag" in tagNames else "threeTag"
        _savefig(fig, var, kwargs.get("outputFolder"), yearName, cut, tagName, region)

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
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    if len(cfg.hists) > 1:
        input_data = cfg.hists
    else:
        input_data = cfg.hists[0]

    hist_to_plot = input_data['hists'][var]
    varName = hist_to_plot.axes[-1].name
    rebin = kwargs.get("rebin", 1)
    plotConfig = cfg.plotConfig
    codes = plotConfig["codes"]

    cut_dict = get_cut_dict(cut, cfg.cutList)

    #
    #  Get the year
    #    (Got to be a better way to do this....)
    #
    yearStr = get_value_nested_dict(cfg.plotConfig, "year", default="RunII")
    year = sum if yearStr == "RunII" else yearStr

    #
    #  Unstacked hists
    #
    process_config = get_value_nested_dict(plotConfig, process)

    tagName = process_config.get("tag", "fourTag")
    tag = plotConfig["codes"]["tag"][tagName]
    # labels.append(v.get("label"))
    # hist_types. append(v.get("histtype", "errorbar"))

    # region_selection = sum if region in ["sum", sum] else hist.loc(codes["region"][region])

    if region in ["sum",sum]:
        region_selection = sum
    elif type(codes["region"][region]) is list:
        region_selection = [hist.loc(_r) for _r in codes["region"][region]]
    else:
        region_selection = hist.loc(codes["region"][region])



    if kwargs.get("debug", False):
        print(f" hist process={process}, "
              f"tag={tag}, year={year}")

    hist_dict = {"process": process_config["process"],
                 "year":    year,
                 "tag":     hist.loc(tag),
                 "region":  region_selection,
                 varName:   hist.rebin(rebin)}


    hist_dict = hist_dict | cut_dict
    _hist = hist_to_plot[hist_dict]


    if len(_hist.shape) == 3:  ## for 2D plots
        _hist = _hist[sum,:,:]
    #
    # Add args
    #
    kwargs["year"] = yearStr

    #
    # Make the plot
    #
    fig, ax = _plot2d(_hist, plotConfig, **kwargs)

    #
    # Save Fig
    #
    if kwargs.get("outputFolder", None):
        _savefig(fig, var, kwargs.get("outputFolder"),
                 yearStr, cut, tagName, region, process)

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

    parser.add_argument('--doTest', action="store_true", help='Metadata file.')
    parser.add_argument('--debug', action="store_true", help='')
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
            #print(f"Adding cut\t{axisName}")
            cutList.append(axisName)
            continue

        if a.extent > 20:
            continue   # HACK to skip the variable bins FIX

        axisLabels[axisName] = []
        #print(axisName)

        for iBin in range(a.extent):

            if axisName in plotConfig["codes"]:
                value = plotConfig["codes"][axisName][a.value(iBin)]
            else:
                value = a.value(iBin)

            #print(f"\t{value}")
            axisLabels[axisName].append(value)

    return axisLabels, cutList

def print_cfg(cfg):
    print("Regions...")
    for reg in cfg.plotConfig["codes"]["region"].keys():
        if type(reg) == str:
            print(f"\t{reg}")

    print("Cuts...")
    for c in cfg.cutList:
        print(f"\t{c}")

    print("Processes...")
    for key, values in cfg.plotConfig.items():
        if key in ["hists", "stack"]:
            for _key, _ in values.items():
                print(f"\t{_key}")
