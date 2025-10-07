import yaml

def load_config_4b(metadata):
    """  Load meta data
    """
    plotConfig = yaml.safe_load(open(metadata, 'r'))

    # for backwards compatibility
    if "codes" not in plotConfig:
        plotConfig['codes'] = {
            'region' : {
                'SR': 2,
                'SB': 1,
                'other': 0,
                2: 'SR',
                1: 'SB',
                0: 'other',
            },
            'tag' : {
                'threeTag': 3,
                'fourTag': 4,
                'other': 0,
                3: 'threeTag',
                4: 'fourTag',
                0: 'other',
            },
        }


    #
    # Expand
    #
    proc_templates = []
    for _hist_proc, _hist_proc_config in plotConfig["hists"].items():
        if not _hist_proc.find("XXX") == -1 and "nSamples" in _hist_proc_config:
            proc_templates.append(_hist_proc)

    for template in proc_templates:
        _hist_proc_config = plotConfig["hists"][template]

        for nS in range(_hist_proc_config["nSamples"]):
            proc_name = template.replace("XXX",str(nS))
            plotConfig["hists"][proc_name] = copy.deepcopy(_hist_proc_config)
            plotConfig["hists"][proc_name]["process"]  = proc_name
            plotConfig["hists"][proc_name]["label"]  = plotConfig["hists"][proc_name]["label"].replace("XXX", str(nS))
            plotConfig["hists"][proc_name]["fillcolor"]  = plot_helpers.COLORS[nS]
            plotConfig["hists"][proc_name]["edgecolor"]  = plot_helpers.COLORS[nS]

        plotConfig["hists"].pop(template)


    return plotConfig
