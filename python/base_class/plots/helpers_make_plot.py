import matplotlib.pyplot as plt
import mplhep as hep  # HEP (CMS) extensions/styling on top of mpl
plt.style.use([hep.style.CMS, {'font.size': 16}])
import matplotlib.patches as mpatches
import base_class.plots.helpers as plot_helpers
import hist
import numpy as np



def plot_leadst_lines():

    def func4(x):
        return (360/x) - 0.5

    def func6(x):
        return max(1.5, (650/x) + 0.5)

    # Plot func4 as a line plot
    x_func4 = np.linspace(100, 1100, 50)
    y_func4 = func4(x_func4)
    plt.plot(x_func4, y_func4, color='red', linestyle='-', linewidth=2)

    # Plot func6 as a line plot
    x_func6 = np.linspace(100, 1100, 50)
    y_func6 = [func6(x) for x in x_func6]
    plt.plot(x_func6, y_func6, color='red', linestyle='-', linewidth=2)

def plot_sublst_lines():

    def func4(x):
        return (235/x)

    def func6(x):
        return max(1.5, (650/x) + 0.7)

    # Plot func4 as a line plot
    x_func4 = np.linspace(100, 1100, 50)
    y_func4 = func4(x_func4)
    plt.plot(x_func4, y_func4, color='red', linestyle='-', linewidth=2)

    # Plot func6 as a line plot
    x_func6 = np.linspace(100, 1100, 50)
    y_func6 = [func6(x) for x in x_func6]
    plt.plot(x_func6, y_func6, color='red', linestyle='-', linewidth=2)


def plot_border_SR():
    # Define the function
    def func0(x, y):
        return (((x - 127.5) / (0.1 * x)) ** 2 + ((y - 122.5) / (0.1 * y)) ** 2)

    def func1(x, y):
        return (((x - 127.5) / (0.1 * x)) ** 2 + ((y - 89.18) / (0.1 * y)) ** 2)

    def func2(x, y):
        return (((x - 92.82) / (0.1 * x)) ** 2 + ((y - 122.5) / (0.1 * y)) ** 2)

    def func3(x, y):
        return (((x - 92.82) / (0.1 * x)) ** 2 + ((y - 89.18) / (0.1 * y)) ** 2)

    # Create a grid of x and y values
    x = np.linspace(0, 250, 500)
    y = np.linspace(0, 250, 500)
    X, Y = np.meshgrid(x, y)

    # Compute the function values on the grid
    Z0 = func0(X, Y)
    Z1 = func1(X, Y)
    Z2 = func2(X, Y)
    Z3 = func3(X, Y)

    # Create the plot
    plt.contour(X, Y, Z0, levels=[1.90*1.90], colors='orangered', linestyles='dashed', linewidths=5)
    plt.contour(X, Y, Z1, levels=[1.90*1.90], colors='orangered', linestyles='dashed', linewidths=5)
    plt.contour(X, Y, Z2, levels=[1.90*1.90], colors='orangered', linestyles='dashed', linewidths=5)
    plt.contour(X, Y, Z3, levels=[2.60*2.60], colors='orangered', linestyles='dashed', linewidths=5)


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

    stack_dict_for_hist = {}
    for k, v in stack_dict.items():
        stack_dict_for_hist[k] = plot_helpers.make_hist(edges=v["edges"],
                                                        values=v["values"],
                                                        variances=v["variances"],
                                                        x_label=v["x_label"],
                                                        under_flow=v["under_flow"],
                                                        over_flow=v["over_flow"],
                                                        add_flow=kwargs.get("add_flow", False)
                                                        )

    #stack_dict_for_hist = {k: v[0] for k, v in stack_dict.items() }
    stack_colors_fill   = [ v.get("fillcolor") for _, v in stack_dict.items() ]
    stack_colors_edge   = [ v.get("edgecolor") for _, v in stack_dict.items() ]

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
    for _, stack_proc_data in stack_dict.items():
        #proc_config = proc_data[1]
        _label = stack_proc_data.get('label')

        if _label in ["None"]:
            continue

        stack_patches.append(mpatches.Patch(facecolor=stack_proc_data.get("fillcolor"),
                                            edgecolor=stack_proc_data.get("edgecolor"),
                                            label    =_label))

    #
    #  Draw the hists
    #
    hist_artists = []
    hist_objs = []
    for hist_proc_name, hist_data in plot_data["hists"].items():

        hist_obj = plot_helpers.make_hist(edges=hist_data["edges"],
                                          values=hist_data["values"],
                                          variances=hist_data["variances"],
                                          x_label=hist_data["x_label"],
                                          under_flow=hist_data["under_flow"],
                                          over_flow=hist_data["over_flow"],
                                          add_flow=kwargs.get("add_flow", False))

        _plot_options = {"density":  norm,
                         "label":    hist_data.get("label", ""),
                         "color":    hist_data.get('fillcolor', 'k'),
                         "histtype": kwargs.get("histtype", hist_data.get("histtype", "errorbar")),
                         "linewidth": kwargs.get("linewidth", hist_data.get("linewidth", 2)),
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

    year_str = plot_helpers.get_year_str(year = kwargs.get('year',"RunII"))

    hep.cms.label("Internal", data=True,
                  year=year_str, loc=0, ax=main_ax)

    main_ax.set_title(f"{plot_helpers.get_region_str(plot_data['region'])}")

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
            linewidth=2.0
        )

        for ratio_name, ratio_data in plot_data["ratio"].items():

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
                                     facecolor=ratio_data.get("facecolor",'none'),
                                     linewidth=0.0,
                                     zorder=1
                                     )

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



def make_plot_from_dict(plot_data, *, do2d=False):
    kwargs = plot_data["kwargs"]
    if do2d:
        fig, ax = _plot2d_from_dict(plot_data, **kwargs)
    else:
        fig, main_ax, ratio_ax = _plot_from_dict(plot_data, **kwargs)
        ax = (main_ax, ratio_ax)

    if kwargs.get("outputFolder", None):

        if type(plot_data.get("process","")) is list:
            tagName = "_vs_".join(plot_data["process"])
        else:
            try:
                tagName = plot_helpers.get_value_nested_dict(plot_data,"tag")
                if isinstance(tagName, hist.loc):
                    tagName = str(tagName.value)
            except ValueError:
                pass

        # these get combined with "/"
        try:
            output_path = [kwargs.get("outputFolder"), kwargs.get("year","RunII"), plot_data["cut"], tagName, plot_data["region"], plot_data.get("process","")]
        except NameError:
            output_path = [kwargs.get("outputFolder")]
        file_name = plot_data.get("file_name",plot_data["var"])
        if kwargs.get("yscale", None) == "log":
            file_name += "_logy"
        plot_helpers.savefig(fig, file_name, *output_path)

        if kwargs.get("write_yaml", False):
            plot_helpers.save_yaml(plot_data, file_name, *output_path)

    return fig, ax



def _plot2d_from_dict(plot_data, **kwargs):

    if kwargs.get("debug", False):
        print(f'\t in plot ... kwargs = {kwargs}')


    if len(plot_data["ratio"]):

        #
        # Plot ratios
        #
        key_iter = iter(plot_data["hists"])
        num_key = next(key_iter)
        num_hist_data = plot_data["hists"][num_key]

        den_key = next(key_iter)
        den_hist_data = plot_data["hists"][den_key]

        ratio_key = next(iter(plot_data["ratio"]))

        # Mask 0s
        hd = np.array(plot_data["ratio"][ratio_key]["ratio"])
        hd[hd < 0.001] = np.nan

        hist_obj_2d = plot_helpers.make_2d_hist(x_edges=num_hist_data["x_edges"], y_edges=num_hist_data["y_edges"],
                                                values=hd,   variances=num_hist_data["variances"],
                                                x_label=num_hist_data["x_label"], y_label=num_hist_data["y_label"])

        scale = 2
        fig = plt.figure(figsize=(10*scale, 6*scale))
        gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.4)
        ax_big = fig.add_subplot(gs[:, 0])
        #fig = plt.figure()   # figsize=(size,size/_phi))
        #fig.add_axes((0.1, 0.15, 0.85, 0.8))
        hist_obj_2d.plot2d(cmap="turbo", cmin=kwargs.get("rlim",[None,None])[0], cmax=kwargs.get("rlim",[None,None])[1])


        ax_top_right = fig.add_subplot(gs[0, 1])

        num_hd = np.array(num_hist_data["values"])
        num_hd[num_hd < 0.001] = np.nan

        num_hist_obj_2d = plot_helpers.make_2d_hist(x_edges=num_hist_data["x_edges"], y_edges=num_hist_data["y_edges"],
                                                    values=num_hd,   variances=num_hist_data["variances"],
                                                    x_label=num_hist_data["x_label"], y_label=num_hist_data["y_label"],
                                                    )

        num_hist_obj_2d.plot2d(cmap="turbo")

        ax_bottom_right = fig.add_subplot(gs[1, 1])
        den_hd = np.array(den_hist_data["values"])
        den_hd[den_hd < 0.001] = np.nan

        den_hist_obj_2d = plot_helpers.make_2d_hist(x_edges=den_hist_data["x_edges"], y_edges=den_hist_data["y_edges"],
                                                    values=den_hd,   variances=den_hist_data["variances"],
                                                    x_label=den_hist_data["x_label"], y_label=den_hist_data["y_label"])

        den_hist_obj_2d.plot2d(cmap="turbo")
        #plt.tight_layout()

    else:

        if len(plot_data["hists"]):
            key = next(iter(plot_data["hists"]))
            hist_data = plot_data["hists"][key]
        elif len(plot_data["stack"]):
            key = next(iter(plot_data["stack"]))
            hist_data = plot_data["stack"][key]
        else:
            raise ValueError(f"ERROR {process} not in plot_data")

        if kwargs.get("full", False):
            hist_obj_2d = plot_helpers.make_2d_hist(x_edges=hist_data["x_edges"], y_edges=hist_data["y_edges"],
                                                    values=hist_data["values"],   variances=hist_data["variances"],
                                                    x_label=hist_data["x_label"], y_label=hist_data["y_label"])

            fig = plt.figure()   # figsize=(size,size/_phi))
            #fig.add_axes((0.1, 0.15, 0.85, 0.8))

            # https://github.com/scikit-hep/hist/blob/main/src/hist/plot.py
            val = hist_obj_2d.plot2d_full(
                main_cmap="jet",
                #top_ls="--",
                top_color="k",
                top_lw=2,
                #side_ls=":",
                side_lw=2,
                side_color="k",
            )
        else:
            # Mask 0s
            hd = np.array(hist_data["values"])
            hd[hd < 0.001] = np.nan

            hist_obj_2d = plot_helpers.make_2d_hist(x_edges=hist_data["x_edges"], y_edges=hist_data["y_edges"],
                                                    values=hd,   variances=hist_data["variances"],
                                                    x_label=hist_data["x_label"], y_label=hist_data["y_label"])


            fig = plt.figure()   # figsize=(size,size/_phi))
            fig.add_axes((0.1, 0.15, 0.85, 0.8))
            hist_obj_2d.plot2d(cmap="turbo")

        if kwargs.get("plot_contour", False): plot_border_SR()
        if kwargs.get("plot_leadst_lines", False): plot_leadst_lines()
        if kwargs.get("plot_sublst_lines", False): plot_sublst_lines()

    ax = fig.gca()

    hep.cms.label("Internal", data=True,
                  year=kwargs.get('year',"RunII").replace("UL", "20"), loc=0, ax=ax)
    ax.set_title(f"{plot_data['region']}  ({plot_data['cut']})", fontsize=16)

    return fig, ax
