import os
import hist
from hist.intervals import ratio_uncertainty
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
plt.style.use([hep.style.CMS, {'font.size': 16}])

_phi = (1 + np.sqrt(5)) / 2
_epsilon = 0.001


def getHist(var='selJets.pt'):
    if var.find("*") != -1:
        ls(match=var.replace("*", ""))
        return

    return hists['hists'][var]


def getFromNestedDict(nested_dict, target_key, default=None):
    """ Return the first value from mathching key from nested dict
    """
    for k, v in nested_dict.items():
        if k == target_key:
            return v

    return default


def _draw_plot(hist_list, stack_dict,  **kwargs):
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
               density=norm)

        s.plot(stack=True, histtype="step",
               color=kwargs.get("stack_colors_edge"),
               density=norm)

    #
    #  Draw the hists
    #
    hist_labels = kwargs.get("hist_labels")
    hist_fill_colors = kwargs.get("hist_fill_colors")
    hist_edge_colors = kwargs.get("hist_edge_colors")
    hist_types = kwargs.get("hist_types")
    for ih, h in enumerate(hist_list):
        this_plot_options = {"density": norm,
                             "label": hist_labels[ih],
                             "color": hist_fill_colors[ih],
                             "histtype": hist_types[ih],
                             "yerr": False,
                             }
        if hist_types[ih] in ["errorbar"]:
            this_plot_options["markersize"] = 7
            this_plot_options["yerr"] = True

        h  .plot(**this_plot_options)

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

    if kwargs.get('legend', False):
        plt.legend(
            # loc='best',      # Position of the legend
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

    # ax.spines["top"]  .set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"] .set_visible(False)

    return fig


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

    _draw_plot(hist_list, stack_dict,  **kwargs)

    top_xlabel = plt.gca().get_xlabel()
    plt.xlabel("")

    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)

    numValues = hist_list[0].values()
    hStackHists = list(stack_dict.values())
    denValues = [h.values() for h in hStackHists]
    denValues = np.sum(denValues, axis=0)

    denValues[denValues == 0] = _epsilon
    ratios = numValues / denValues

    if kwargs.get("norm",  False):
        numSF = np.sum(hist_list[0].values(), axis=0)
        denSF = np.sum(denValues, axis=0)
        ratios *= denSF/numSF

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
    #  ylabel
    #
    plt.ylabel(kwargs.get("rlabel", "Ratio"))
    plt.ylabel(plt.gca().get_ylabel(), loc='center')

    plt.xlabel(kwargs.get("xlabel", top_xlabel), loc='right')

    plt.ylim(*kwargs.get('rlim', [0, 2]))

    return fig


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

    h = hists['hists'][var]
    varName = hists['hists'][var].axes[-1].name
    rebin = kwargs.get("rebin", 1)
    codes = plotConfig["codes"]

    cutDict = {}
    for c in cutList:
        cutDict[c] = sum
    cutDict[cut] = True

    #
    #  Get the year
    #    (Got to be a better way to do this....)
    #
    year = "RunII"
    tagName = "fourTag"
    if len(plotConfig["hists"]):
        year = getFromNestedDict(plotConfig["hists"], "year", "RunII")
        tag = getFromNestedDict(plotConfig["hists"], "tag",  "fourTag")
    else:
        year = getFromNestedDict(plotConfig["stack"], "year", "RunII")
        tag = getFromNestedDict(plotConfig["stack"], "tag",  "fourTag")

    print(f"year is {year}")

    #
    #  Unstack hists
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
        this_tag = plotConfig["codes"]["tag"][v["tag"]]
        hist_colors_fill.append(v.get('fillcolor'))
        hist_colors_edge.append(v.get('edgecolor'))
        hist_labels.append(v.get("label"))
        hist_types. append(v.get("histtype", "errorbar"))

        if kwargs.get("debug", False):
            print(f" hist process={this_process}, "
                  f"tag={this_tag}, year={this_year}")

        this_hist_dict = {"process": this_process,  "year": this_year,
                          "tag": hist.loc(this_tag),
                          "region": hist.loc(codes["region"][region]),
                          varName: hist.rebin(rebin)}

        for c in cutList:
            this_hist_dict = this_hist_dict | {c: cutDict[c]}

        hists.append(h[this_hist_dict])
        hists[-1] *= v.get("scalefactor", 1.0)

    #
    # Add args
    #
    kwargs["year"] = year
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

    for k, v in stack_config.items():
        this_year2 = sum if v["year"] == "RunII" else v["year"]
        this_tag2 = plotConfig["codes"]["tag"][v["tag"]]
        this_process = v['process']
        this_label = v.get('label')
        stack_colors_fill.append(v.get('fillcolor'))
        stack_colors_edge.append(v.get('edgecolor'))

        if kwargs.get("debug", False):
            print("Drawing")
            print(f" stack_config process={this_process},"
                  f"tag={this_tag2}, year={this_year2}")

        this_hist_opts = {"process": this_process,  "year": this_year2,
                          "tag": hist.loc(this_tag2),
                          "region": hist.loc(codes["region"][region]),
                          varName: hist.rebin(rebin)}
        for c in cutList:
            this_hist_opts = this_hist_opts | {c: cutDict[c]}

        stack_dict[this_label] = h[this_hist_opts]

    #
    # Pass colors
    #
    kwargs["stack_colors_fill"] = stack_colors_fill
    kwargs["stack_colors_edge"] = stack_colors_edge

    if kwargs.get("doRatio", False):
        fig = _plot_ratio(hists, stack_dict, plotConfig, **kwargs)
    else:
        fig = _plot(hists, stack_dict, plotConfig,  **kwargs)

    if kwargs.get("outputFolder", None):
        yearName = year
        outputPath = kwargs.get("outputFolder")+"/"
        outputPath += "/".join([yearName, cut, tagName, region])

        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        fig.savefig(outputPath+"/"+var.replace(".", '_')+".pdf")

    return fig
