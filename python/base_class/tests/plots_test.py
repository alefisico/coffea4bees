import unittest
from coffea.util import load
import yaml
import sys
import os
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import get_value_nested_dict, makePlot, load_config, load_hists, read_axes_and_cuts
import sys
import base_class.plots.iPlot_config as cfg
import numpy as np
from base_class.tests.parser import wrapper

#
# python3 analysis/tests/plot_test.py   --inputFile analysis/hists/test.coffea --knownCounts base_class/tests/plotCounts.yml 
#

class PlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        #self.plotsAllDict = yaml.safe_load(open(, 'r'))
        metadata = "analysis/metadata/plotsAll.yml"
        #inputFile = "analysis/hists/test.coffea"
        inputFile = wrapper.args["inputFile"]
        
        cfg.plotConfig = load_config(metadata)
        cfg.hists = load_hists([inputFile])
        cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

        #  Make these numbers with:
        #  >  python     base_class/tests/dumpPlotCounts.py --input [inputFileName] -o [outputFielName]
        #       (python base_class/tests/dumpPlotCounts.py --input analysis/hists/test.coffea --output base_class/tests/testPlotCounts.yml)
        #
        knownCountFile = wrapper.args["knownCounts"] 
        self.knownCounts = yaml.safe_load(open(knownCountFile, 'r'))
        

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

        for k, v  in self.knownCounts.items():
            print(f"testing...{k}")
            var = v["var"]
            cut = v["cut"]
            region = v["region"]
            counts = v["counts"]

            fig, ax = makePlot(cfg.hists[0], cfg.cutList, cfg.plotConfig,
                               var=var, cut=cut, region=region,
                               outputFolder=cfg.outputFolder, **default_args)

            y_plot = ax.lines[-1].get_ydata()
            np.testing.assert_array_equal(y_plot, counts)


if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)

