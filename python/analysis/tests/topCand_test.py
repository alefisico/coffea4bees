import unittest
import argparse
from coffea.util import load
import yaml
from parser import wrapper
import sys

import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import time

import os
sys.path.insert(0, os.getcwd())
from analysis.helpers.topCandReconstruction import buildTops_single, find_tops, find_tops_slow


class topCandRecoTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.inputFile = wrapper.args["inputFile"]
        #
        # Test vectors from picos
        #
        self.input_jet_pt = [[243, 223, 71.8, 67.8], [181, 119, 116, 96.9], [208, 189, 62.4, 52.1], [296, 157, 143, 125, 69.3], [118, 64.6, 58.6, 44.1]]
        self.input_jet_eta = [[1.51, 0.516, 0.656, 0.99], [2.02, 0.795, 0.747, 1.45], [1.09, -0.219, 1.5, 0.66], [1.63, -0.4, -1.44, 1.03, -1.04], [0.802, 1.48, 1.56, -0.822]]
        self.input_jet_phi = [[-2.88, 0.5, -1.88, 0.118], [-0.834, 1.87, 2.71, 1.8], [2.21, -0.736, -2.53, 2.12], [-2.26, 0.32, 1.2, -2.66, 0.862], [-2.84, 0.558, 2.06, -0.639]]
        self.input_jet_mass = [[28.9, 30.8, 9.66, 13.2], [19.7, 22.8, 15.8, 12.2], [25.8, 24.7, 8.77, 7.87], [23.9, 17.4, 22.9, 11.8, 11.4], [19, 13.1, 10, 7.66] ]
        self.input_jet_btagDeepFlavB = [[0.999, 0.907, 0.0116, 0.685], [0.603, 1, 0.997, 0.646], [0.985, 0.975, 0.0113, 0.967], [0.0131, 0.302, 0.936, 0.102, 0.977],[0.999, 0.00486, 0.764, 1]]
        self.input_jet_bRegCorr = [[1.1, 1.12, 1.07, 1.02], [0.962, 1, 1.18, 0.977], [1.01, 1.01, 1.09, 1.29], [1, 1.04, 1.05, 1.07, 1.03], [1.03, 1.04, 1.18, 0.961]]
        self.output_xbW =  [5.1162261962890625, 3.6441168785095215, 0.7684999108314514, 4.558335304260254, 5.11277961730957]
        self.output_xW = [3.4038572311401367, 7.014884948730469, 1.7819664478302002, 8.788503646850586, 0.7852322459220886]



    def test_topCand_single(self):
        print("test_topCand_single\n\n")

        for idx in range(len(self.output_xW)):
            #
            #  make coffea objects
            #
            input_jets = ak.zip(
                {
                    "pt":  self.input_jet_pt[idx],
                    "eta": self.input_jet_eta[idx],
                    "phi": self.input_jet_phi[idx],
                    "mass": self.input_jet_mass[idx],
                    "btagDeepFlavB": self.input_jet_btagDeepFlavB[idx],
                    "bRegCorr": self.input_jet_bRegCorr[idx],
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )
    
            input_jets = input_jets[ak.argsort(input_jets.btagDeepFlavB, axis=0, ascending=False)]
            xWs, xbWs, xWbWs = buildTops_single(input_jets)
            min_xWbW = np.argmin(xWbWs)
            #print(f"xWs is {xWs} vs {self.output_xW[idx]}")
            #print(f"xbWs is {xbWs} vs {self.output_xbW[idx]}")
            #print(f"xWbW is {xWbWs} ")
            xW_min = xWs[min_xWbW]
            xbW_min = xbWs[min_xWbW]
            print(f"xW_min is {xW_min} vs {self.output_xW[idx]}  diff {xW_min - self.output_xW[idx]}   ({(xW_min - self.output_xW[idx])/self.output_xW[idx]})")
            print(f"xbW_min is {xbW_min} vs {self.output_xbW[idx]} diff {xbW_min - self.output_xbW[idx]} ({(xbW_min - self.output_xbW[idx])/self.output_xbW[idx]})")
            

    def test_topCand(self):
        print("test_topCand\n\n")
        input_jets = ak.zip(
            {
                "pt": self.input_jet_pt,
                "eta": self.input_jet_eta,
                "phi": self.input_jet_phi,
                "mass": self.input_jet_mass,
                "btagDeepFlavB": self.input_jet_btagDeepFlavB,
                "bRegCorr": self.input_jet_bRegCorr,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        print(input_jets)
        input_jets = input_jets[ak.argsort(input_jets.btagDeepFlavB, axis=1, ascending=False)]
        print(input_jets)


        start = time.perf_counter()
        top_cands = find_tops_slow(input_jets)
        end = time.perf_counter()
        print("Bare Python = {}s".format((end - start)))
        print(top_cands)

        start = time.perf_counter()
        top_cands = find_tops(input_jets)
        end = time.perf_counter()
        print("Elapsed (with compilation) = {}s".format((end - start)))
        print(top_cands)

        start = time.perf_counter()
        top_cands = find_tops(input_jets)
        end = time.perf_counter()
        print("Elapsed (with compilation) = {}s".format((end - start)))
        print(top_cands)

        top_cands = [input_jets[top_cands[idx]] for idx in "012"]
        print(top_cands)


        rec_top_cands = ak.zip({
            "w": ak.zip({
                "jl": top_cands[1] + top_cands[2],
            }),
            "bReg": top_cands[0]* top_cands[0].bRegCorr
        })

        mW =  80.4;
        #xW = (dijet_jl.mass-mW)/(0.10*dijet_jl.mass) 
        #print(rec_top_cands)
        #print(rec_top_cands.w)
        #print(rec_top_cands.w.jl)
        rec_top_cands["xW"] = (rec_top_cands.w.jl.mass- mW)/(0.10*rec_top_cands.w.jl.mass) 
        rec_top_cands["w", "jl_wCor"] = rec_top_cands.w.jl * (mW/rec_top_cands.w.jl.mass)
        #dijet_jl_wCor  = dijet_jl*(mW/dijet_jl.mass);
        rec_top_cands["mbW"] = (rec_top_cands.bReg + rec_top_cands.w.jl_wCor).mass
        mt = 173.0;
        rec_top_cands["xbW"]  = (rec_top_cands.mbW-mt)/(0.05*rec_top_cands.mbW) #smaller resolution term because there are fewer degrees of freedom. FWHM=25GeV, about the same as mW 
        rec_top_cands["xWbW"] = np.sqrt( rec_top_cands.xW ** 2 + rec_top_cands.xbW**2)
        print(rec_top_cands.xW)        
        print(rec_top_cands.xbW)        
        print(rec_top_cands.xWbW)        

        rec_top_cands = rec_top_cands[ak.argsort(rec_top_cands.xWbW, axis=1, ascending=True)]
        print("Post sort")
        print(rec_top_cands.xW)        
        print(rec_top_cands.xbW)        
        print(rec_top_cands.xWbW)        
        xW_min = rec_top_cands[:, 0].xW
        xbW_min = rec_top_cands[:, 0].xbW


        print(f"xW_min is {xW_min} vs {self.output_xW}  diff {xW_min - self.output_xW}   ({(xW_min - self.output_xW)/self.output_xW})")
        print(f"xbW_min is {xbW_min} vs {self.output_xbW} diff {xbW_min - self.output_xbW} ({(xbW_min - self.output_xbW)/self.output_xbW})")

        
        #print(rec_top_cands.xW)        
        #top_cands[1] + top_cands[2],
        

#
        #xWs, xbWs, xWbWs = buildTops(input_jets)

    


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
