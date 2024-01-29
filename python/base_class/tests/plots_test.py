import unittest
import argparse
from coffea.util import load
import yaml
import sys
import os
sys.path.insert(0, os.getcwd())
from base_class.plots import get_value_nested_dict
import sys

class PlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.plotsAllDict = yaml.safe_load(open("analysis/metadata/plotsAll.yml", 'r'))


    def test_get_value_nested_dict(self):
        
        testDict = {"dA0" : {"dA1" : {"dA2": {"ka0" : "va0", "ka1" : "va1"} }, "ka2" : "va2"},
                    "dB0" : {"kb0" : "vb0"},
                    "kb1" : "vb1"}
            
        self.assertEqual(get_value_nested_dict(testDict, "ka0"), "va0")
        self.assertEqual(get_value_nested_dict(testDict, "ka1"), "va1")
        self.assertEqual(get_value_nested_dict(testDict, "ka2"), "va2")
        self.assertEqual(get_value_nested_dict(testDict, "kb0"), "vb0")
        self.assertEqual(get_value_nested_dict(testDict, "kb1"), "vb1")

        
        value = get_value_nested_dict(self.plotsAllDict["hists"], "year")
        self.assertEqual(value, "RunII")

        value = get_value_nested_dict(self.plotsAllDict, "year")
        self.assertEqual(value, "RunII")        

        value = get_value_nested_dict(self.plotsAllDict, "fillcolor")
        self.assertEqual(value, "k")        


        
                

if __name__ == '__main__':
    unittest.main(argv=sys.argv)

