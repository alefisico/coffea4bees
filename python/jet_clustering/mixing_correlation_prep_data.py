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
import pickle


sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.helpers_make_plot_dict as plot_helpers_make_plot_dict
import base_class.plots.helpers_make_plot as plot_helpers_make_plot
import base_class.plots.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')



def main():

    input_file_SvB   = cfg.hists[0]
    input_file_hData = cfg.hists[1]

    # 'event', 'run', 'luminosityBlock', 'SvB_MA_ps'
    vX = [f"v{i}" for i in range(15)]
    eras = ["2016","2017","2018"]

    #
    #  Start with 2 vs and one era
    #
    vX   = [f"v{i}" for i in range(15)]
    #eras = ["2016"]

    #
    #  Construct mapping (event, run) -> hemi data
    #
    event_to_hemi_data = {}
    for _vX in vX:

        for _era in eras:
            _key = f"mix_{_vX}_{_era}"

            nEvents = len(input_file_hData["event"][_key])
            for _iE in range(nEvents):

                event  = input_file_hData["event"]          [_key][_iE]
                run    = input_file_hData["run"]            [_key][_iE]

                h1_event  = input_file_hData["h1_event"]          [_key][_iE]
                h1_run    = input_file_hData["h1_run"]            [_key][_iE]

                h2_event  = input_file_hData["h2_event"]          [_key][_iE]
                h2_run    = input_file_hData["h2_run"]            [_key][_iE]


                event_id = (event, run)

                #if event_id in hemi_data_map:
                #    print(f"ERROR repeated event {event_id}")
                #    print(f"\t old {hemi_data_map[event_id]}")
                #    print(f"\t new {( (h1_event, h1_run), (h2_event, h2_run) )}")

                event_to_hemi_data[event_id] =  ( (h1_event, h1_run), (h2_event, h2_run) )



    #
    # combine data by event/run/LB
    #
    SvB_per_event = {}
    output_file_SvB = {}
    output_file_SvB["h1_run"]    = {}
    output_file_SvB["h1_event"]  = {}
    output_file_SvB["h1_weights"]  = {}
    output_file_SvB["h2_run"]    = {}
    output_file_SvB["h2_event"]  = {}
    output_file_SvB["h2_weights"]  = {}
    output_file_SvB["SvB_MA_ps"] = {}
    output_file_SvB["event"]     = {}
    output_file_SvB["run"]       = {}

    N_toy = 30
    hemi_data_to_weights = {}

    for _vX in vX:

        for _era in eras:

            _key = f"mix_{_vX}_{_era}"
            #_key = f"syn_v0_{_era}"
            print(_key)

            output_file_SvB["h1_run"]   [_key] = []
            output_file_SvB["h1_event"] [_key] = []
            output_file_SvB["h1_weights"] [_key] = []
            output_file_SvB["h2_run"]   [_key] = []
            output_file_SvB["h2_event"] [_key] = []
            output_file_SvB["h2_weights"] [_key] = []
            output_file_SvB["SvB_MA_ps"][_key] = []
            output_file_SvB["event"]    [_key] = []
            output_file_SvB["run"]      [_key] = []


            nEvents = len(input_file_SvB["event"][_key])
            for _iE in range(nEvents):
                SvB    = input_file_SvB["SvB_MA_ps"]      [_key][_iE]
                if SvB < 0.001: continue

                event  = input_file_SvB["event"]          [_key][_iE]
                run    = input_file_SvB["run"]            [_key][_iE]

                # Remove TTbar
                if run == 1:
                    continue

                event_id = (event, run)
                if event_id not in event_to_hemi_data:
                    print(f"ERROR event missing {event_id}")

                h1_event_id, h2_event_id = event_to_hemi_data[event_id]

                if h1_event_id not in hemi_data_to_weights:
                    hemi_data_to_weights[h1_event_id] = np.random.poisson(lam=1, size=(N_toy))

                if h2_event_id not in hemi_data_to_weights:
                    hemi_data_to_weights[h2_event_id] = np.random.poisson(lam=1, size=(N_toy))

                output_file_SvB["SvB_MA_ps"][_key].append(SvB)
                output_file_SvB["event"]    [_key].append(event)
                output_file_SvB["run"]      [_key].append(run)

                output_file_SvB["h1_run"]    [_key].append( h1_event_id[1] )
                output_file_SvB["h1_event"]  [_key].append( h1_event_id[0] )
                output_file_SvB["h1_weights"][_key].append( hemi_data_to_weights[h1_event_id] )

                output_file_SvB["h2_run"]    [_key].append( h2_event_id[1] )
                output_file_SvB["h2_event"]  [_key].append( h2_event_id[0] )
                output_file_SvB["h2_weights"][_key].append( hemi_data_to_weights[h2_event_id] )



    #
    # Write out pikle
    #
    with open('merged_mixedData_SvB.pkl', 'wb') as f:
        pickle.dump(output_file_SvB, f)





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
