import unittest
import sys
import os
#sys.path.insert(0, os.getcwd())

#from base_class.JCMTools import ncr  # Replace 'your_module' with the name of your Python file without the .py extension
#
#class TestNCR(unittest.TestCase):
#    def test_known_values(self):
#        self.assertEqual(ncr(5, 2), 10)
#        self.assertEqual(ncr(6, 3), 20)
#        self.assertEqual(ncr(10, 0), 1)
#        self.assertEqual(ncr(0, 0), 1)
#
#    def test_symmetry(self):
#        n = 10
#        for r in range(n+1):
#            self.assertEqual(ncr(n, r), ncr(n, n-r))
#
#    def test_negative_values(self):
#        with self.assertRaises(ValueError):  # Assuming you want to raise a ValueError for negative inputs
#            ncr(-1, 5)
#        with self.assertRaises(ValueError):  # Adjust according to the behavior you expect
#            ncr(5, -1)


import yaml

class TestJCM(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        #
        # make output
        #
        #os.system("python analysis/make_weights_ROOT.py -o testJCM_ROOT_tests -c passPreSel -r SB --ROOTInputs")
        #os.system("python analysis/make_weights_ROOT.py -o testJCM_Coffea_tests -c passPreSel -r SB ")

        #
        #  > From python analysis/tests/dumpROOTToHist.py -o analysis/tests/HistsFromROOTFile.coffea -c passPreSel -r SB
        #
        pass

    def test_yaml_content(self):

        for test_pair in [#('analysis/testJCM_ROOT_tests/jetCombinatoricModel_SB_.yml', 'analysis/tests/jetCombinatoricModel_SB_ROOT.yml'),
                          #('analysis/testJCM_Coffea_tests/jetCombinatoricModel_SB_.yml', 'analysis/tests/jetCombinatoricModel_SB_Coffea.yml'),

                          ('analysis/testJCM_ROOT/jetCombinatoricModel_SB_.yml',   'analysis/tests/jetCombinatoricModel_SB_ROOT_new.yml'),
                          ('analysis/testJCM_Coffea/jetCombinatoricModel_SB_.yml', 'analysis/tests/jetCombinatoricModel_SB_Coffea_new.yml'),

                          ]:

            test_file      = test_pair[0]
            reference_file = test_pair[1]
            print("testing",test_file)

            # Load the content of the test YAML file
            with open(test_file, 'r') as file:
                test_data = yaml.safe_load(file)

            # Load the content of the reference YAML file
            with open(reference_file, 'r') as file:
                reference_data = yaml.safe_load(file)

            for k, v in test_data.items():
                if type(v) == list:
                    for listIdx in range(len(v)):
                        self.assertAlmostEqual(v[listIdx], reference_data[k][listIdx], delta=0.001, msg=f"Failed match {k}... {v} vs {reference_data[k]} ... Diff: {v[listIdx]-reference_data[k][listIdx]}")
                else:
                    self.assertAlmostEqual(v, reference_data[k], delta=0.001, msg=f"Failed match {k}... {v} vs {reference_data[k]} ... Diff: {v-reference_data[k]}")


if __name__ == '__main__':
    unittest.main()

