import argparse
from coffea.util import load
import sys
import os
import numpy as np
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import get_value_nested_dict, makePlot, load_config, load_hists, read_axes_and_cuts
import base_class.plots.iPlot_config as cfg


def print_counts_yaml(var, cut, region, counts):

    outputFile.write(f"{'_'.join([var,cut,region])}:\n")
    outputFile.write(f"    var:\n")
    outputFile.write(f"        {var}\n")
    outputFile.write(f"    cut:\n")
    outputFile.write(f"        {cut}\n")
    outputFile.write(f"    region:\n")
    outputFile.write(f"           {region}\n")
    outputFile.write(f"    counts:\n")
    outputFile.write(f"           {counts.tolist()}\n")        
    outputFile.write("\n\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o','--outputFile', default='knownCounts.yml', help='Input File. Default: hists.pkl')
    args = parser.parse_args()

    outputFile = open(f'{args.outputFile}', 'w')

    metadata = "analysis/metadata/plotsAll.yml"
    cfg.plotConfig = load_config(metadata)
    cfg.hists = load_hists([args.inputFile])
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    default_args = {"doRatio":0, "rebin":4, "norm":0}    

    test_vectors = [("SvB_MA.ps", "passPreSel", "SR"),
                    ("SvB_MA.ps", "passPreSel", "SB"),
                    ("SvB_MA.ps", "passSvB",    "SR"),
                    ("SvB_MA.ps", "passSvB",    "SB"),
                    ("SvB_MA.ps", "failSvB",    "SR"),
                    ("SvB_MA.ps", "failSvB",    "SB"),
                    ]
    
    for tv in test_vectors:

        var    = tv[0]
        cut    = tv[1]
        region = tv[2]
        print(f"testing {var}, {cut}, {region}")
        fig, ax = makePlot(cfg.hists[0], cfg.cutList, cfg.plotConfig,
                           var=var, cut=cut, region=region,
                           outputFolder=cfg.outputFolder, **default_args)
        
        counts = ax.lines[-1].get_ydata()

        print_counts_yaml(var, cut, region, counts)




