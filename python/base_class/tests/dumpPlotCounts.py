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
    #for k in sorted(counts.keys(),reverse=True):
    #
    #    outputFile.write(f"    {k}:\n")
    #    for cut in ['passJetMult', 'passPreSel', 'passDiJetMass', 'SR', 'SB', 'passSvB', 'failSvB']:
    #        try:
    #            outputFile.write(f"        {cut}: {round(float(counts[k][cut]),2)}\n")
    #        except KeyError: pass
    #    outputFile.write("\n")
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

    test_vectors = [("SvB_MA.ps","passPreSel","SR", np.array([10.,  4.,  6.,  4.,  3.,  5.,  6.,  1.,  4.,  5.,  2.,  5.,  3.,
                                                              4.,  1.,  2.,  3.,  4.,  3.,  6.,  2.,  1.,  0.,  1.,  0.])
                     ),
                    ("SvB_MA.ps","passPreSel", "SB", np.array([64., 37., 24.,  9.,  9.,  7.,  2.,  5.,  1.,  1.,  3.,  0.,  0.,
                                                               0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                     ),
                    ("SvB_MA.ps","passSvB", "SR", np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                            0., 0., 0., 2., 1., 0., 1., 0.])
                     ),
                    ("SvB_MA.ps","passSvB", "SB", np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                            0., 0., 0., 0., 0., 0., 0., 0.])
                     ),
                    ("SvB_MA.ps","failSvB", "SR", np.array([10.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                     ),
                    ("SvB_MA.ps","failSvB", "SB", np.array([64., 17.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                                                            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
                     ),
                    ]
    
    for tv in test_vectors:

        var    = tv[0]
        cut    = tv[1]
        region = tv[2]
        print(f"testing {var}, {cut}, {region}")
        fig, ax = makePlot(cfg.hists[0], cfg.cutList, cfg.plotConfig,
                           var=var, cut=cut, region=region,
                           outputFolder=cfg.outputFolder, **default_args)
        
        #y_plot = ax.lines[-1].get_ydata()
        counts = ax.lines[-1].get_ydata()

        print_counts_yaml(var, cut, region, counts)
#    print_counts_yaml(cf3, "counts3")
#    print_counts_yaml(cf4_unit, "counts4_unit")
#    print_counts_yaml(cf3_unit, "counts3_unit")




