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


def _savefig(fig, var, *args):
    outputPath = "/".join(args)

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    fig.savefig(outputPath + "/" + var.replace(".", '_') + ".pdf")
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


def _plot_ratio(hist_list, stack_dict, plotConfig, **kwargs):
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

    numValues = hist_list[0].values()

    #
    # If stack is given use it for the denominator
    #   else, take the last histogram
    #
    if stack_dict:
        hStackHists = list(stack_dict.values())
        denValues = [h.values() for h in hStackHists]
        denValues = np.sum(denValues, axis=0)
    else:
        denValues = hist_list[-1].values()

    denValues[denValues == 0] = _epsilon
    ratios = numValues / denValues

    if kwargs.get("norm", False):
        numSF = np.sum(hist_list[0].values(), axis=0)
        denSF = np.sum(denValues, axis=0)
        ratios *= denSF / numSF

    # Set 0 and inf to nan to hide during plotting
    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan

    central_value_artist = subplot_ax.axhline(
        kwargs.get("ratio_line_value", 1.0),
        color="black",
        linestyle="dashed",
        linewidth=1.0
    )

    # https://github.com/scikit-hep/hist/blob/main/src/hist/intervals.py
    ratio_uncert = ratio_uncertainty(
        num=numValues,
        denom=denValues,
        uncertainty_type=kwargs.get("uncertainty_type", "poisson"),
    )

    x_values = hist_list[0].axes[0].centers

    subplot_ax.errorbar(
        x_values,
        ratios,
        yerr=ratio_uncert,
        color="black",
        marker="o",
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


def _makeHistsFromList(input_hist_File, cutList, plotConfig, var, cut, region, process, **kwargs):

    if kwargs.get("debug", False):
        print(f" hist process={process}, "
              f"cut={cut}")

    rebin = kwargs.get("rebin", 1)
    codes = plotConfig["codes"]

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
        cut_dict = get_cut_dict(cut, cutList)

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

            this_cut_dict = get_cut_dict(_cut, cutList)

            this_hist_dict = process_dict | tag_dict | region_dict | year_dict | var_dict | this_cut_dict

            hists.append(input_hist_File['hists'][var][this_hist_dict])
            hists[-1] *= process_config.get("scalefactor", 1.0)

    #
    #  region list
    #
    elif type(region) is list:

        for ir, _reg in enumerate(region):

            if kwargs.get("debug", False):
                print_list_debug_info(process, this_tag, cut, _reg)

            hist_colors_fill.append(_colors[ir])
            hist_labels.append(label + " " + _reg)
            hist_types. append("errorbar")

            this_region_dict = {"region": hist.loc(codes["region"][_reg])}

            this_hist_dict = process_dict | tag_dict | this_region_dict | year_dict | var_dict | cut_dict

            hists.append(input_hist_File['hists'][var][this_hist_dict])
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
            hists.append(input_hist_File[iF]['hists'][var][this_hist_dict])
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

            hists.append(input_hist_File['hists'][var][this_hist_dict])
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

            hists.append(input_hist_File['hists'][_var][this_hist_dict])
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
        fig, main_ax, ratio_ax = _plot_ratio(hists, {}, plotConfig, **kwargs)
        ax = (main_ax, ratio_ax)
    else:
        fig, ax = _plot(hists, {}, plotConfig, **kwargs)

    if kwargs.get("outputFolder", None):
        _savefig(fig, var, kwargs.get("outputFolder"), yearStr, cutName, this_tagName, regionName, process)

    return fig, ax


def makePlot(hists, cutList, plotConfig, var='selJets.pt',
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

    if (type(cut) is list) or (type(region) is list) or (type(hists) is list) \
       or (type(process) is list) or (type(var) is list):
        return _makeHistsFromList(hists, cutList, plotConfig, var, cut, region, **kwargs)

    if process and type(process) is not list:
        process = [process]
        kwargs["process"] = process
        return _makeHistsFromList(hists, cutList, plotConfig, var, cut, region, **kwargs)

    h = hists['hists'][var]
    varName = hists['hists'][var].axes[-1].name
    rebin = kwargs.get("rebin", 1)
    var_dict = {varName: hist.rebin(rebin)}

    codes = plotConfig["codes"]

    if cut not in cutList:
        raise AttributeError(f"{cut} not in cutList {cutList}")

    cut_dict = get_cut_dict(cut, cutList)
    region_dict = {"region":  hist.loc(codes["region"][region])}

    tagNames = []

    #
    #  Unstacked hists
    #
    hists = []
    hist_config = plotConfig["hists"]
    hist_colors_fill = []
    hist_colors_edge = []
    hist_labels = []
    hist_types = []

    for k, v in hist_config.items():
        this_process = v["process"]
        this_year = sum if v["year"] == "RunII" else v["year"]
        tagNames.append(v["tag"])
        this_tag = plotConfig["codes"]["tag"][v["tag"]]
        hist_colors_fill.append(v.get('fillcolor'))
        hist_colors_edge.append(v.get('edgecolor'))
        hist_labels.append(v.get("label"))
        hist_types. append(v.get("histtype", "errorbar"))

        if kwargs.get("debug", False):
            print(f" hist process={this_process}, "
                  f"tag={this_tag}, year={this_year}")

        this_hist_dict = {"process": this_process,
                          "year":    this_year,
                          "tag":     hist.loc(this_tag),
                          }

        this_hist_dict = this_hist_dict | var_dict | region_dict | cut_dict

        hists.append(h[this_hist_dict])
        hists[-1] *= v.get("scalefactor", 1.0)

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
    stack_colors_fill = []
    stack_colors_edge = []
    stack_labels = []

    for k, v in stack_config.items():
        this_year = sum if v["year"] == "RunII" else v["year"]
        year_dict = {"year": this_year}

        stack_labels.append(v.get('label'))
        stack_colors_fill.append(v.get('fillcolor'))
        stack_colors_edge.append(v.get('edgecolor'))

        if v.get("process", None):
            this_process = v['process']
            tagNames.append(v["tag"])

            this_tag = plotConfig["codes"]["tag"][v["tag"]]

            if kwargs.get("debug", False):
                print("Drawing")
                print(f" stack_config process={this_process},"
                      f"tag={this_tag}, year={this_year}")

            this_hist_opts = {"process": this_process,
                              "tag": hist.loc(this_tag),
                              }

            this_hist_opts = this_hist_opts | var_dict | region_dict | year_dict | cut_dict

            stack_dict[k] = h[this_hist_opts]

        elif v.get("sum", None):

            hist_sum = None
            for sum_k, sum_v in v.get("sum").items():

                this_process = sum_v['process']
                tagNames.append(sum_v["tag"])
                this_tag = plotConfig["codes"]["tag"][sum_v["tag"]]

                this_hist_opts = {"process": this_process,
                                  "tag": hist.loc(this_tag),
                                  }

                this_hist_opts = this_hist_opts | var_dict | region_dict | year_dict | cut_dict

                this_hist = h[this_hist_opts]
                this_hist *= sum_v.get("scalefactor", 1.0)
                if hist_sum:
                    hist_sum += this_hist
                else:
                    hist_sum = this_hist

            stack_dict[k] = hist_sum

        else:
            raise Exception("Error need to config either process or sum")
    #
    # Pass colors
    #
    kwargs["stack_colors_fill"] = stack_colors_fill
    kwargs["stack_colors_edge"] = stack_colors_edge
    kwargs["stack_labels"]      = stack_labels

    if kwargs.get("doRatio", False):
        fig, main_ax, ratio_ax = _plot_ratio(hists, stack_dict, plotConfig, **kwargs)
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


def make2DPlot(hists, process, cutList, plotConfig, var='selJets.pt',
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

    h = hists['hists'][var]
    varName = hists['hists'][var].axes[-1].name
    rebin = kwargs.get("rebin", 1)
    codes = plotConfig["codes"]

    cut_dict = get_cut_dict(cut, cutList)

    #
    #  Get the year
    #    (Got to be a better way to do this....)
    #
    yearStr = get_value_nested_dict(plotConfig, "year", default="RunII")
    year = sum if yearStr == "RunII" else yearStr

    #
    #  Unstacked hists
    #
    process_config = get_value_nested_dict(plotConfig, process)

    tagName = process_config.get("tag", "fourTag")
    tag = plotConfig["codes"]["tag"][tagName]
    # labels.append(v.get("label"))
    # hist_types. append(v.get("histtype", "errorbar"))

    region_selection = sum if region in ["sum", sum] else hist.loc(codes["region"][region])

    if kwargs.get("debug", False):
        print(f" hist process={process}, "
              f"tag={tag}, year={year}")

    hist_dict = {"process": process_config["process"],
                 "year":    year,
                 "tag":     hist.loc(tag),
                 "region":  region_selection,
                 varName:   hist.rebin(rebin)}

    hist_dict = hist_dict | cut_dict

    _hist = h[hist_dict]

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

    parser = argparse.ArgumentParser(description='uproot_plots')

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
            print(f"Adding cut\t{axisName}")
            cutList.append(axisName)
            continue

        if a.extent > 20:
            continue   # HACK to skip the variable bins FIX

        axisLabels[axisName] = []
        print(axisName)

        for iBin in range(a.extent):

            if axisName in plotConfig["codes"]:
                value = plotConfig["codes"][axisName][a.value(iBin)]
            else:
                value = a.value(iBin)

            print(f"\t{value}")
            axisLabels[axisName].append(value)

    return axisLabels, cutList
