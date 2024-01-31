import unittest
import argparse
from coffea.util import load
import yaml
from parser import wrapper
import sys

import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector

import os
sys.path.insert(0, os.getcwd())
from analysis.helpers.topCandReconstruction import buildTops


class iPlotTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.inputFile = wrapper.args["inputFile"]


    def test_topCand(self):
        #
        # Test vectors from picos
        #
        input_jet_pt = [[243, 223, 71.8, 67.8], [181, 119, 116, 96.9], [208, 189, 62.4, 52.1]]
        input_jet_eta = [[1.51, 0.516, 0.656, 0.99], [2.02, 0.795, 0.747, 1.45], [1.09, -0.219, 1.5, 0.66]]
        input_jet_phi = [[-2.88, 0.5, -1.88, 0.118], [-0.834, 1.87, 2.71, 1.8], [2.21, -0.736, -2.53, 2.12]]
        input_jet_mass = [[28.9, 30.8, 9.66, 13.2], [19.7, 22.8, 15.8, 12.2], [25.8, 24.7, 8.77, 7.87]]
        input_jet_btagDeepFlavB = [[0.999, 0.907, 0.0116, 0.685], [0.603, 1, 0.997, 0.646], [0.985, 0.975, 0.0113, 0.967]]
        input_jet_bRegCorr = [[1.1, 1.12, 1.07, 1.02], [0.962, 1, 1.18, 0.977], [1.01, 1.01, 1.09, 1.29]]
        output_xbW =  [5.1162261962890625, 3.6441168785095215, 0.7684999108314514]
        output_xW = [3.4038572311401367, 7.014884948730469, 1.7819664478302002]

        idx = 2

        #
        #  make coffea objects
        #
        input_jets = ak.zip(
            {
                "pt": input_jet_pt[idx],
                "eta": input_jet_eta[idx],
                "phi": input_jet_phi[idx],
                "mass": input_jet_mass[idx],
                "btagDeepFlavB": input_jet_btagDeepFlavB[idx],
                "bRegCorr": input_jet_bRegCorr[idx],
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        input_jets = input_jets[ak.argsort(input_jets.btagDeepFlavB, axis=0, ascending=False)]
        xWs, xbWs = buildTops(input_jets)
        print(f"xW is {xWs}")
        print(f"xbW is {xbWs}")

# From https://github.com/patrickbryant/ZZ4b/blob/master/nTupleAnalysis/src/eventData.cc#L1333
# for(auto &b: topQuarkBJets){
#    for(auto &j: topQuarkWJets){
#      if(b.get()==j.get()) continue; //require they are different jets
#      if(b->deepFlavB < j->deepFlavB) continue; //don't consider W pairs where j is more b-like than b.
#      for(auto &l: topQuarkWJets){
#      if(b.get()==l.get()) continue; //require they are different jets
#      if(j.get()==l.get()) continue; //require they are different jets
#      if(j->deepFlavB < l->deepFlavB) continue; //don't consider W pairs where l is more b-like than j.
#            trijet* thisTop = new trijet(b,j,l);
#            if(thisTop->xWbW < xWbW0){
#                      xWt0 = thisTop->xWt;
#                      xWbW0= thisTop->xWbW;
#                      dRbW = thisTop->dRbW;
#                      t0.reset(thisTop);
#                      xWt = xWt0; // define global xWt in this case
#                      xWbW= xWbW0;
#                      xW = thisTop->W->xW;
#                      xt = thisTop->xt;
#                      xbW = thisTop->xbW;
#                      t = t0;
#                    }else{delete thisTop;}
#      }
#    }
#  }
#  if(nSelJets<5) return; 



        
        

if __name__ == '__main__':
    wrapper.parse_args()
    unittest.main(argv=sys.argv)
