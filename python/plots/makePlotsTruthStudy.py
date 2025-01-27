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


def doPlots(debug=False):

    #
    #  Truth Study Plots
    #
    plot(["v4j40.mass","v4j15.mass","v4j20.mass","v4j30.mass","v4j00.mass"], region="SR", cut="pass4GenBJets00", process="HH4b",    doRatio=1, ylim=[0,20000],rlim=[0,1],
         labels=["b-jet $p_{T}$ > 40 GeV", "b-jet $p_{T}$ > 15 GeV", "b-jet $p_{T}$ > 20 GeV", "b-jet $p_{T}$ > 30 GeV", "b-jet $p_{T}$ > 00 GeV",], histtype="step", xlim=[100,1000])
    plot(["genBJet0.pt","genBJet1.pt","genBJet2.pt","genBJet3.pt"], region="SR", cut="pass4GenBJets00", process="HH4b",    doRatio=0, xlim=[0,400], xlabel="b-jet Gen $p_T$",labels=["leading","2nd","3rd","4th"],histtype="step")
    plot(["genBJet0.pt","genBJet1.pt","genBJet2.pt","genBJet3.pt"], region="SR", cut="pass4GenBJets00", process="HH4b",    doRatio=0, xlim=[0,100], xlabel="b-jet Gen $p_T$",labels=["leading","2nd","3rd","4th"],histtype="step")

    plot("otherGenJet00.pt", region="SR", cut="pass4GenBJets00", process="HH4b",    doRatio=0,yscale="log", xlabel="leading non-bJet $p_T$")


    plot( "v4j20.mass", region="SR", cut=["pass4GenBJetsb203b40_1j_e","pass4GenBJets40"], process="HH4b",    doRatio=1, rlim=[0,1],histtype="step", xlim=[100,1000],labels=["b20 3b40 1j40","4b40"])

    plot( "v4j20.mass", region="SR", cut=["pass4GenBJetsb203b40_1j_e","pass4GenBJets2b202b40_2j_e","pass4GenBJets40"], process="HH4b",    doRatio=1, rlim=[0,1],histtype="step", xlim=[100,1000],labels=["b20 3b40 1j40","2b20 2b40 2j40","4b40"])

    plot( "v4j20.pt", region="SR", cut=["pass4GenBJetsb203b40_1j_e","pass4GenBJets40"], process="HH4b",    doRatio=1, rlim=[0,1],histtype="step", xlim=[0,400],labels=["b20 3b40 1j40","4b40"])


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
