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
import yaml

sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args
import base_class.plots.iPlot_config as cfg

np.seterr(divide='ignore', invalid='ignore')

def plot(var, **kwargs):
    fig, ax = makePlot(cfg, var, outputFolder= args.outputFolder, **kwargs) 
    plt.close()
    return fig, ax


def doPlots(debug=False):

    # Fixes Needed !!!
    plot("canJet3.pt", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("canJets.eta", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("quadJet_min_dr.close.dr", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"],xlim=[0,2])

    
    #
    #  Jet Level
    #
    plot("canJets.pt",  region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("canJets.eta", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("canJets.phi", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("canJets.energy", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("canJets.mass", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("canJets.pz", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])

    #
    #  Di-Jet Level
    #
    quad_jets = ["quadJet_min_dr","quadJet_selected"]
    di_jets = ["lead","subl","close","other"]

    for q in quad_jets:
        for d in di_jets:
            plot(f"{q}.{d}.dphi", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
            plot(f"{q}.{d}.dr", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
            plot(f"{q}.{d}.eta", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
            plot(f"{q}.{d}.mass", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
            plot(f"{q}.{d}.phi", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
            plot(f"{q}.{d}.pt", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
            plot(f"{q}.{d}.pz", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
            #plot(f"{q}.{d}.dphi", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])


    plot("v4j.pt", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("v4j.phi", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("v4j.eta", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    plot("v4j.mass", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    
    #
    #  Event Level
    #
    plot("SvB_MA.ps", region="SB", cut="passPreSel",doRatio=1,rebin=1,process="data",histtype="step",debug=0,norm=1,labels=["De-clustered","Nominal"])
    

        
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
    
    #varList = [ h for h in cfg.hists[0]['hists'].keys() if not h in args.skip_hists ]
    doPlots(debug=args.debug)
