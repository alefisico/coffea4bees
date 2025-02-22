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

sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')


def plot(var, **kwargs):
    makePlot(cfg, var, outputFolder= args.outputFolder, **kwargs)
    plt.close()

def doPlots(varList, debug=False):

    # Fig 32
    # TTbar wrong here...
    plot("selJets_noJCM.n",region="SB",yscale="linear",norm=1,rebin=1,doratio=1,rlim=[0,2],xlim=[4,15])
    plot("tagJets_noJCM.n",region="SB",yscale="linear",norm=1,rebin=1,doratio=1,rlim=[0,2],xlim=[0,15])

    # Fig 34
    plot("selJets.n",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0,2],xlim=[4,15])


    # Fig 43
    plot("quadJet_min_dr.close.dr",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("quadJet_min_dr.other.dr",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])

    # Fig 44
    plot("canJets.pt",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("canJets.eta",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5],xlim=[-3,3])
    plot("canJets.phi",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5],xlim=[-0.2,3.2])
    plot("canJets.mass",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5],xlim=[0,50])

    # Fig 45
    plot("othJets.pt",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("othJets.eta",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("othJets.phi",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("othJets.mass",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5],xlim=[0,50])

    # Fig 46
    plot("quadJet_selected.lead.mass",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("quadJet_selected.subl.mass",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("quadJet_selected.lead.pt",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5],xlim=[0,500])
    plot("quadJet_selected.subl.pt",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5],xlim=[0,500])

    # Fig 47
    plot("v4j.mass",region="SB",yscale="linear",norm=0,rebin=2,doratio=1,rlim=[0.5,1.5])
    plot("v4j.pt",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("v4j.pz",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])
    plot("quadJet_selected.dr",region="SB",yscale="linear",norm=0,rebin=1,doratio=1,rlim=[0.5,1.5])

    # Fig 48
    plot("SvB_MA_noFvT.ps_zz",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_MA.ps_zz",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_MA_noFvT.ps_zh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_MA.ps_zh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_MA_noFvT.ps_hh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_MA.ps_hh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])

    # Fig 49
    plot("SvB_noFvT.ps_zz",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB.ps_zz",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_noFvT.ps_zh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB.ps_zh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_noFvT.ps_hh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB.ps_hh",region="SB",yscale="log",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])



    # Fig 98
    plot("SvB_MA.ps_zz",region="SR",yscale="linear",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB.ps_zz",   region="SR",yscale="linear",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_MA.ps_zh",region="SR",yscale="linear",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB.ps_zh",   region="SR",yscale="linear",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB_MA.ps_hh",region="SR",yscale="linear",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])
    plot("SvB.ps_hh",   region="SR",yscale="linear",norm=0,rebin=8,doratio=1,rlim=[0.5,1.5])


    #
    # L2 REview
    #
    plot("quadJet_selected.subl.mass",region=["SB","SR"],yscale="linear",rebin=1,doratio=0,process="Multijet",histtype="step")
    plot("quadJet_selected.lead.mass",region=["SB","SR"],yscale="linear",rebin=1,doratio=0,process="Multijet",histtype="step")

    plot("quadJet_selected.subl.mass",region=["SB","SR"],yscale="linear",rebin=1,doratio=0,process="TTbar",histtype="step")
    plot("quadJet_selected.lead.mass",region=["SB","SR"],yscale="linear",rebin=1,doratio=0,process="TTbar",histtype="step")






if __name__ == '__main__':

    args = parse_args()

    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder

    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    varList = [ h for h in cfg.hists[0]['hists'].keys() if not h in args.skip_hists ]
    doPlots(varList, debug=args.debug)
