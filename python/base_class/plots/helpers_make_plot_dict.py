import copy
import base_class.plots.helpers as plot_helpers
import hist
import numpy as np


def print_list_debug_info(process, tag, cut, region):
    print(f" hist process={process}, "
          f"tag={tag}, _cut={cut}"
          f"_reg={region}")


#
#  Get hist values
#
def get_hist_data(*, process, cfg, config, var, region, cut, rebin, year, do2d=False, file_index=None, debug=False):

    if year in  ["RunII", "Run2", "Run3", "RunIII"]:
        year     = sum

    if debug:
        print(f" hist process={process}, "
              f"tag={config.get('tag', None)}, year={year}, var={var}")

    hist_opts = {"process": process,
                 "year":  year,
                 "tag":   config.get("tag", None),
                 "region": region
                 }

    if region == "sum":
        hist_opts["region"] = sum

    cut_dict = plot_helpers.get_cut_dict(cut, cfg.cutList)

    hist_opts = hist_opts | cut_dict

    hist_obj = None
    if len(cfg.hists) > 1 and not cfg.combine_input_files:
        if file_index is None:
            print("ERROR must give file_index if running with more than one input file without using the  --combine_input_files option")

        common, unique_to_dict = plot_helpers.compare_dict_keys_with_list(hist_opts, cfg.hists[file_index]['categories'])

        if len(unique_to_dict) > 0:
            for _key in unique_to_dict:
                hist_opts.pop(_key)

        hist_obj = cfg.hists[file_index]['hists'][var]

        if "variation" in cfg.hists[file_index]["categories"]:
            hist_opts = hist_opts | {"variation" : "nominal"}

    else:
        for _input_data in cfg.hists:

            common, unique_to_dict = plot_helpers.compare_dict_keys_with_list(hist_opts, _input_data['categories'])

            if len(unique_to_dict) > 0:
                for _key in unique_to_dict:
                    hist_opts.pop(_key)

            if var in _input_data['hists'] and process in _input_data['hists'][var].axes["process"]:

                if "variation" in _input_data["categories"]:
                    hist_opts = hist_opts | {"variation" : "nominal"}

                hist_obj = _input_data['hists'][var]

    if hist_obj is None:
        raise ValueError(f"ERROR did not find var {var} with process {process} in inputs")

    ## for backwards compatibility
    for axis in hist_obj.axes:
        if (axis.name == "tag") and isinstance(axis, hist.axis.IntCategory):
            hist_opts['tag'] = hist.loc(cfg.plotConfig["codes"]["tag"][config["tag"]])
        if (axis.name == "region") and isinstance(axis, hist.axis.IntCategory):
            if isinstance(hist_opts['region'], list):
                hist_opts['region'] = [ hist.loc(cfg.plotConfig["codes"]["region"][i]) for i in hist_dict['region'] ]
            elif region != "sum":
                hist_opts['region'] = hist.loc(cfg.plotConfig["codes"]["region"][region])

    #
    #  Add rebin Options
    #
    varName = hist_obj.axes[-1].name
    if not do2d:
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
    if do2d:
        if len(selected_hist.shape) == 3:  # for 2D plots
            selected_hist = selected_hist[sum, :, :]
    else:
        if len(selected_hist.shape) == 2:
            selected_hist = selected_hist[sum, :]

    #
    # Apply Scale factor
    #
    selected_hist *= config.get("scalefactor", 1.0)

    return selected_hist



#
def get_hist_data_list(*, proc_list, cfg, config, var, region, cut, rebin, year, do2d, file_index, debug):

    selected_hist = None
    for _proc in proc_list:

        if type(_proc) is list:
            _selected_hist =  get_hist_data_list(proc_list=_proc, cfg=cfg, config=config, var=var, region=region,
                                                 cut=cut, rebin=rebin, year=year, do2d=do2d, file_index=file_index, debug=debug)
        else:
            _selected_hist = get_hist_data(process=_proc, cfg=cfg, config=config, var=var, region=region,
                                           cut=cut, rebin=rebin, year=year, do2d=do2d, file_index=file_index, debug=debug)

        if selected_hist is None:
            selected_hist = _selected_hist
        else:
            selected_hist += _selected_hist

    return selected_hist


#
#  Get hist from input file(s)
#
def add_hist_data(*, cfg, config, var, region, cut, rebin, year, do2d=False, file_index=None, debug=False):

    if debug:
        print(f"In add_hist_data {config['process']} \n")

    proc_list = config['process'] if type(config['process']) is list else [config['process']]

    selected_hist = get_hist_data_list(proc_list=proc_list, cfg=cfg, config=config, var=var, region=region,
                                       cut=cut, rebin=rebin, year=year, do2d=do2d, file_index=file_index, debug=debug)

    if do2d:

        # Extract counts and variances
        try:
            config["values"]    = selected_hist.view(flow=False)["value"].tolist()  # Bin counts (array)
            config["variances"] = selected_hist.view(flow=False)["variance"].tolist()  # Bin variances (array)
        except IndexError:
            config["values"]    = selected_hist.values()  # Bin counts (array)
            config["variances"] = selected_hist.variances()  # Bin variances (array)
        if config["variances"] is None:
            config["variances"] = np.zeros_like(config["values"])

        config["x_edges"]   = selected_hist.axes[0].edges.tolist()  # X-axis edges
        config["y_edges"]   = selected_hist.axes[1].edges.tolist()  # Y-axis edges
        config["x_label"]   = selected_hist.axes[0].label  # X-axis label
        config["y_label"]   = selected_hist.axes[1].label  # Y-axis label

    else:
        config["values"]     = selected_hist.values().tolist()
        config["variances"]  = selected_hist.variances().tolist()
        config["centers"]    = selected_hist.axes[0].centers.tolist()
        config["edges"]      = selected_hist.axes[0].edges.tolist()
        config["x_label"]    = selected_hist.axes[0].label
        config["under_flow"] = float(selected_hist.view(flow=True)["value"][0])
        config["over_flow"]  = float(selected_hist.view(flow=True)["value"][-1])

    return



def get_plot_dict_from_list(*, cfg, var, cut, region, process, **kwargs):

    if kwargs.get("debug", False):
        print(f" in _makeHistFromList hist process={process}, "
              f"cut={cut}")

    rebin = kwargs.get("rebin", 1)
    do2d = kwargs.get("do2d", False)
    var_over_ride = kwargs.get("var_over_ride", {})
    label_override = kwargs.get("labels", None)
    year = kwargs.get("year", "RunII")

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
    plot_data["kwargs"] = kwargs
    plot_data["process"] = process


    #
    #  Parse the Lists
    #
    if type(process) is list:
        process_config = [plot_helpers.get_value_nested_dict(cfg.plotConfig, p) for p in process]
    else:
        try:
            process_config = plot_helpers.get_value_nested_dict(cfg.plotConfig, process)

            proc_id = process_config["label"] if type(process_config["process"]) is list else process_config["process"]

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
            _process_config["fillcolor"] = plot_helpers.colors[ic]
            _process_config["label"]     = plot_helpers.get_label(f"{process_config['label']} { _cut}", label_override, ic)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            add_hist_data(cfg=cfg, config=_process_config,
                          var=var_to_plot, region=region, cut=_cut, rebin=rebin, year=year,
                          do2d=do2d,
                          debug=kwargs.get("debug", False))

            plot_data["hists"][proc_id + _cut + str(ic)] = _process_config

    #
    #  region list
    #
    elif type(region) is list:
        for ir, _reg in enumerate(region):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), cut, _reg)

            _process_config = copy.deepcopy(process_config)
            _process_config["fillcolor"] = plot_helpers.colors[ir]
            _process_config["label"]     = plot_helpers.get_label(f"{process_config['label']} { _reg}", label_override, ir)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            add_hist_data(cfg=cfg, config=_process_config,
                          var=var_to_plot, region=_reg, cut=cut, rebin=rebin, year=year,
                          do2d = do2d,
                          debug=kwargs.get("debug", False))
            plot_data["hists"][proc_id + _reg + str(ir)] = _process_config


    #
    #  input file list
    #
    elif len(cfg.hists) > 1 and not cfg.combine_input_files:
        if kwargs.get("debug", False):
            print_list_debug_info(process, process_config.get("tag"), cut, region)

        fileLabels = kwargs.get("fileLabels", [])

        for iF, _input_File in enumerate(cfg.hists):

            _process_config = copy.deepcopy(process_config)
            _process_config["fillcolor"] = plot_helpers.colors[iF]

            if label_override:
                _process_config["label"] = label_override[iF]
            elif iF < len(fileLabels):
                _process_config["label"] = _process_config["label"] + " " + fileLabels[iF]
            else:
                _process_config["label"] = _process_config["label"] + " file" + str(iF + 1)

            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            add_hist_data(cfg=cfg, config=_process_config,
                          var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                          do2d=do2d, file_index=iF,
                          debug=kwargs.get("debug", False))

            plot_data["hists"][proc_id + "file" + str(iF)] = _process_config

    #
    #  process list
    #
    elif type(process) is list:
        for iP, _proc_conf in enumerate(process_config):

            if kwargs.get("debug", False):
                print_list_debug_info(_proc_conf["process"], _proc_conf.get("tag"), cut, region)

            _process_config = copy.deepcopy(_proc_conf)
            _process_config["fillcolor"] = _proc_conf.get("fillcolor", None)#.replace("yellow", "orange")
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            _proc_id = _proc_conf["label"] if type(_proc_conf["process"]) is list else _proc_conf["process"]

            var_to_plot = var_over_ride.get(_proc_id, var)

            add_hist_data(cfg=cfg, config=_process_config,
                          var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                          do2d=do2d,
                          debug=kwargs.get("debug", False))

            plot_data["hists"][_proc_id + str(iP)] = _process_config



    #
    #  var list
    #
    elif type(var) is list:
        for iv, _var in enumerate(var):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), cut, region)

            _process_config = copy.deepcopy(process_config)
            _process_config["fillcolor"] = plot_helpers.colors[iv]
            _process_config["label"]     = plot_helpers.get_label(f"{process_config['label']} { _var}", label_override, iv)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            add_hist_data(cfg=cfg, config=_process_config,
                          var=_var, region=region, cut=cut, rebin=rebin, year=year,
                          do2d=do2d,
                          debug=kwargs.get("debug", False))
            plot_data["hists"][proc_id + _var + str(iv)] = _process_config


    #
    #  year list
    #
    elif type(year) is list:
        for iy, _year in enumerate(year):

            if kwargs.get("debug", False):
                print_list_debug_info(process, process_config.get("tag"), cut, region)

            _process_config = copy.copy(process_config)
            _process_config["fillcolor"] = plot_helpers.colors[iy]
            _process_config["label"]     = plot_helpers.get_label(f"{process_config['label']} { _year}", label_override, iy)
            _process_config["histtype"]  = kwargs.get("histtype","errorbar")

            add_hist_data(cfg=cfg, config=_process_config,
                          var=var, region=region, cut=cut, rebin=rebin, year=_year,
                          do2d=do2d,
                          debug=kwargs.get("debug", False))
            plot_data["hists"][proc_id + _year + str(iy)] = _process_config


    else:
        raise Exception("Error something needs to be a list!")



    if kwargs.get("doRatio", kwargs.get("doratio", False)):

        if do2d:
            hist_keys = list(plot_data["hists"].keys())
            den_key = hist_keys.pop(0)

            denValues  = np.array(plot_data["hists"][den_key]["values"])
            denVars    = plot_data["hists"][den_key]["variances"]
            denValues[denValues == 0] = plot_helpers.epsilon

            #
            #  for 2d only do one ratio
            #
            num_key = hist_keys.pop(0)
            numValues  = np.array(plot_data["hists"][num_key]["values"])
            numVars    = plot_data["hists"][num_key]["variances"]

            ratio_config = {}
            ratios, ratio_uncert = plot_helpers.makeRatio(numValues, numVars, denValues, denVars, **kwargs)
            ratio_config["ratio"] = ratios.tolist()
            ratio_config["error"] = ratio_uncert.tolist()
            plot_data["ratio"][f"ratio_{num_key}_to_{den_key}"] = ratio_config


        else:

            hist_keys = list(plot_data["hists"].keys())
            den_key = hist_keys.pop(0)

            denValues  = np.array(plot_data["hists"][den_key]["values"])
            denVars    = plot_data["hists"][den_key]["variances"]
            denCenters = plot_data["hists"][den_key]["centers"]

            denValues[denValues == 0] = plot_helpers.epsilon

            # Bkg error band

            band_ratios = np.ones(len(denCenters))
            band_uncert  = np.sqrt(denVars * np.power(denValues, -2.0))
            band_config = {"color": "k",  "type": "band", "hatch": "\\\\",
                           "ratio":band_ratios.tolist(),
                           "error":band_uncert.tolist(),
                           "centers": list(denCenters)}
            plot_data["ratio"]["bkg_band"] = band_config

            for iH, _num_key in enumerate(hist_keys):

                numValues  = np.array(plot_data["hists"][_num_key]["values"])
                numVars    = plot_data["hists"][_num_key]["variances"]

                ratio_config = {"color": plot_helpers.colors[iH],
                                "marker": "o",
                                }
                ratios, ratio_uncert = plot_helpers.makeRatio(numValues, numVars, denValues, denVars, **kwargs)
                ratio_config["ratio"] = ratios.tolist()
                ratio_config["error"] = ratio_uncert.tolist()
                ratio_config["centers"] = denCenters
                plot_data["ratio"][f"ratio_{_num_key}_to_{den_key}_{iH}"] = ratio_config

    return plot_data


def load_stack_config(*, cfg, stack_config, var, cut, region, **kwargs):

    stack_dict = {}
    var_over_ride = kwargs.get("var_over_ride", {})
    rebin   = kwargs.get("rebin", 1)
    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)
    do2d    = kwargs.get("do2d", False)

    #
    #  Loop processes in the stack config
    #
    for _proc_name, _proc_config in stack_config.items():

        proc_config = copy.deepcopy(_proc_config)

        var_to_plot = var_over_ride.get(_proc_name, var)

        if kwargs.get("debug", False):
            print(f"stack_process is {_proc_name} var is {var_to_plot}")

        #
        #  If this component is a process in the hist_obj
        #
        if proc_config.get("process", None):


            #
            #  Get the hist object from the input data file(s)
            #
            add_hist_data(cfg=cfg, config=proc_config,
                          var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                          do2d=do2d,
                          debug=kwargs.get("debug", False))

            stack_dict[_proc_name] = proc_config

        #
        #  If this compoent is a sum of processes in the hist_obj
        #
        elif proc_config.get("sum", None):

            for sum_proc_name, sum_proc_config in proc_config.get("sum").items():

                sum_proc_config["year"] = _proc_config["year"]

                var_to_plot = var_over_ride.get(sum_proc_name, var)

                #
                #  Get the hist object from the input data file(s)
                #
                add_hist_data(cfg=cfg, config=sum_proc_config,
                              var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                              do2d=do2d,
                              debug=kwargs.get("debug", False))



            stack_values = [v["values"] for _, v in proc_config["sum"].items()]
            proc_config["values"] = np.sum(stack_values, axis=0).tolist()

            stack_variances = [v["variances"] for _, v in proc_config["sum"].items()]
            proc_config["variances"] = np.sum(stack_variances, axis=0).tolist()

            first_sum_entry = next(iter(proc_config["sum"].values()))
            proc_config["centers"] = first_sum_entry["centers"]
            proc_config["edges"]   = first_sum_entry["edges"]
            proc_config["x_label"] = first_sum_entry["x_label"]

            stack_under_flow = [v["under_flow"] for _, v in proc_config["sum"].items()]
            proc_config["under_flow"] = float(np.sum(stack_under_flow, axis=0).tolist())

            stack_over_flow = [v["over_flow"] for _, v in proc_config["sum"].items()]
            proc_config["over_flow"] = float(np.sum(stack_over_flow, axis=0))

            stack_dict[_proc_name] = proc_config

        else:
            raise Exception("Error need to config either process or sum")

    return stack_dict



def get_values_variances_centers_from_dict(hist_config, plot_data):

    if hist_config["type"] == "hists":
        num_data = plot_data["hists"][hist_config["key"]]
        return np.array(num_data["values"]), np.array(num_data["variances"]), num_data["centers"]


    if hist_config["type"] == "stack":
        return_values = [v["values"] for _, v in plot_data["stack"].items()]
        return_values = np.sum(return_values, axis=0)

        return_variances = [v["variances"] for _, v in plot_data["stack"].items()]
        return_variances = np.sum(return_variances, axis=0)

        centers = next(iter(plot_data["stack"].values()))["centers"]

        return return_values, return_variances, centers

    raise ValueError("ERROR: ratio needs to be of type 'hists' or 'stack'")



def add_ratio_plots(ratio_config, plot_data, **kwargs):

    for r_name, _r_config in ratio_config.items():

        r_config = copy.deepcopy(_r_config)

        numValues, numVars, numCenters = get_values_variances_centers_from_dict(r_config.get("numerator"),   plot_data)
        denValues, denVars, _          = get_values_variances_centers_from_dict(r_config.get("denominator"), plot_data)

        if kwargs.get("norm", False):
            r_config["norm"] = True

        #
        #  Ratios
        #
        ratios, ratio_uncert = plot_helpers.makeRatio(numValues, numVars, denValues, denVars, **r_config)
        r_config["ratio"]  = ratios.tolist()
        r_config["error"]  = ratio_uncert.tolist()
        r_config["centers"] = numCenters
        plot_data["ratio"][f"ratio_{r_name}"] = r_config

        #
        # Bkg error band
        #
        default_band_config = {"color": "k",  "type": "band", "hatch": "\\\\\\"}
        _band_config = r_config.get("bkg_err_band", default_band_config)

        if _band_config:
            band_config = copy.deepcopy(_band_config)
            band_config["ratio"] = np.ones(len(numCenters)).tolist()
            denValues[denValues == 0] = plot_helpers.epsilon
            band_config["error"] = np.sqrt(denVars * np.power(denValues, -2.0)).tolist()
            band_config["centers"] = list(numCenters)
            plot_data["ratio"][f"band_{r_name}"] = band_config


    return



def get_plot_dict_from_config(*, cfg, var='selJets.pt',
                              cut="passPreSel", region="SR", **kwargs):

    process = kwargs.get("process", None)
    year    = kwargs.get("year", "RunII")
    rebin   = kwargs.get("rebin", 1)
    do2d    = kwargs.get("do2d",  False)
    debug   = kwargs.get("debug", False)

    # Make process a list if it exits and isnt one already
    if process is not None and type(process) is not list:
        process = [process]

    #
    #  Lets you plot different variables for differnet processes
    #
    var_over_ride = kwargs.get("var_over_ride", {})

    if cut and cut not in cfg.cutList:
        raise AttributeError(f"{cut} not in cutList {cfg.cutList}")

    #
    #  Unstacked hists
    #
    plot_data = {}
    plot_data["hists"] = {}
    plot_data["stack"] = {}
    plot_data["ratio"] = {}
    plot_data["var"] = var
    plot_data["cut"] = cut
    plot_data["region"] = region
    if do2d:
        plot_data["process"] = process[0]
        plot_data["is_2d_hist"] = True
    plot_data["kwargs"] = kwargs

    #hists = []
    hist_config = cfg.plotConfig["hists"]

    # for a single process
    if process is not None:
        hist_config = {key: hist_config[key] for key in process if key in hist_config}

    #
    #  Loop of hists in config file
    #
    for _proc_name, _proc_config in hist_config.items():

        proc_config = copy.deepcopy(_proc_config)

        #
        #  Add name to config
        #
        proc_config["name"] = _proc_name  ### REMOVE COMMENT

        var_to_plot = var_over_ride.get(_proc_name, var)

        #
        #  Get the hist object from the input data file(s)
        #
        add_hist_data(cfg=cfg, config=proc_config,
                      var=var_to_plot, region=region, cut=cut, rebin=rebin, year=year,
                      do2d=do2d,
                      debug=kwargs.get("debug", False))
        plot_data["hists"][_proc_name] = proc_config



    #
    #  The stack
    #
    stack_config = cfg.plotConfig.get("stack", {})
    if process is not None:
        stack_config = {key: stack_config[key] for key in process if key in stack_config}

    plot_data["stack"] = load_stack_config(cfg=cfg, stack_config=stack_config, var=var, cut=cut, region=region, **kwargs)


    #
    #  Config Ratios
    #
    if kwargs.get("doRatio", kwargs.get("doratio", False)) and not do2d:
        ratio_config = cfg.plotConfig["ratios"]
        add_ratio_plots(ratio_config, plot_data, **kwargs)

    return plot_data
