import os
import sys
import yaml
import hist
import matplotlib.pyplot as plt
from hist.intervals import ratio_uncertainty
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args, print_cfg
import base_class.plots.iPlot_config as cfg




if __name__ == '__main__':
    args = parse_args()
    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files
    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    print_cfg(cfg)



    import pickle

    vars_to_write = ["pt_comb", "eta_comb", "zA","thetaA","mA", "mB", "decay_phi"]

    cfg.hists[0][vars_to_write[0]]

    keys = cfg.hists[0][vars_to_write[0]].keys()

    data = {}
    for _v in vars_to_write:
        _v_data = []
        for _k in keys:
            _v_data += cfg.hists[0][_v][_k]
        data[_v] = _v_data
    #print(

    print(len(data["pt_comb"]))
    # Write to a binary file using pickle
    with open('splitting_data_bb.pkl', 'wb') as f:
        pickle.dump(data, f)
