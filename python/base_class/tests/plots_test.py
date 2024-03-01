import unittest
import argparse
from coffea.util import load
import yaml
import sys
import os
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import get_value_nested_dict, makePlot, load_config, load_hists, read_axes_and_cuts
import sys
import base_class.plots.iPlot_config as cfg
import numpy as np

class PlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        #self.plotsAllDict = yaml.safe_load(open(, 'r'))
        metadata = "analysis/metadata/plotsAll.yml"
        inputFile = "analysis/hists/test.coffea"

        cfg.plotConfig = load_config(metadata)
        cfg.hists = load_hists([inputFile])
        cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

        

    def test_get_value_nested_dict(self):
        
        testDict = {"dA0" : {"dA1" : {"dA2": {"ka0" : "va0", "ka1" : "va1"} }, "ka2" : "va2"},
                    "dB0" : {"kb0" : "vb0"},
                    "kb1" : "vb1"}
            
        self.assertEqual(get_value_nested_dict(testDict, "ka0"), "va0")
        self.assertEqual(get_value_nested_dict(testDict, "ka1"), "va1")
        self.assertEqual(get_value_nested_dict(testDict, "ka2"), "va2")
        self.assertEqual(get_value_nested_dict(testDict, "kb0"), "vb0")
        self.assertEqual(get_value_nested_dict(testDict, "kb1"), "vb1")

        
        value = get_value_nested_dict(cfg.plotConfig["hists"], "year")
        self.assertEqual(value, "RunII")

        value = get_value_nested_dict(cfg.plotConfig, "year")
        self.assertEqual(value, "RunII")        

        value = get_value_nested_dict(cfg.plotConfig, "fillcolor")
        self.assertEqual(value, "k")        


    def test_counts(self):        

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

            y_plot = ax.lines[-1].get_ydata()
            np.testing.assert_array_equal(y_plot, tv[3])
                

if __name__ == '__main__':
    unittest.main(argv=sys.argv)

