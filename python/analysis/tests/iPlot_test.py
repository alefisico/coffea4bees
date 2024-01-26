import unittest
import argparse
from coffea.util import load
import yaml
from parser import wrapper
import sys


import os
sys.path.insert(0, os.getcwd())

#from iPlot import load_config
from analysis.iPlot import load_config, load_hists, read_axes_and_cuts, plot, plot2d
import analysis.iPlot_config as cfg
        
class iPlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        metadata = "analysis/metadata/plotsAll.yml"
        cfg.plotConfig = load_config(metadata)

        input_files = [wrapper.args["inputFile"]]
        cfg.hists = load_hists(input_files)
            

        cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)


    def test_makePlot(self):
        args = {"var": "v4j.*", "region":"SR", "cut":"passPreSel"}
        doRatio = {"doRatio":1}
        norm  = {"norm":1}
        logy    = {"yscale":"log"}
        rlim    = {"rlim":[0,2]}
        
        print(f"plot with {args}")
        plot(**args)
        args["var"] = "v4j.mass"

        print(f"plot with {args}")
        plot(**args)

        print(f"plot with {args | doRatio}")
        plot(**(args | doRatio))

        print(f"plot with {args | norm}")
        plot(**(args | norm))

        print(f"plot with {args | logy}")
        plot(**(args | logy))

        print(f"plot with {args | doRatio | norm}")
        plot(**(args | doRatio | norm))

        print(f"plot with {args | doRatio | norm | rlim}")
        plot(**(args | doRatio | norm | rlim))

        args2d = {"var": "quadJet_min_dr.close_vs_other_m", "region":"SR", "cut":"passPreSel", "process":"Multijet"}
        full = {"full" : True}
        print(f"plot with {args2d}")
        plot2d(**args2d)

        print(f"plot with {args2d | full}")
        plot2d(**(args2d | full))        

        args["var"] = "v4j.mass"
        invalid_region = {"region":"InvalidRegion"}
        invalid_cut    = {"cut"   :"InvalidCut"}

        print(f"plot with {args | invalid_region}")
        self.assertRaises(KeyError, plot, **(args | invalid_region))

        print(f"plot with {args | invalid_cut}")
        self.assertRaises(AttributeError, plot, **(args | invalid_cut))

        # To Add
        # Different regions
        # Different cuts
        
        
if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)

