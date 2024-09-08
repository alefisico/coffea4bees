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


def plotCut(args):

    # Fixes Needed !!!
    args["rebin"] = 1
    plot("canJet3.pt", **args)
    plot("canJets.eta", **args)
    plot("quadJet_min_dr.close.dr", **args,xlim=[0,2])


    #
    #  Jet Level
    #
    plot("canJets.pt",  **args)
    plot("canJets.eta", **args, xlim=[-3,3])
    plot("canJets.phi", **args)
    plot("canJets.energy", **args)
    plot("canJets.mass", **args)#),xlim=[0,50])
    plot("canJets.pz", **args)#,yscale="log")

    for i in range(4):
        plot(f"canJet{i}.pt",  **args)
        plot(f"canJet{i}.eta", **args, xlim=[-3,3])
        plot(f"canJet{i}.phi", **args)
        plot(f"canJet{i}.energy", **args)
        plot(f"canJet{i}.mass", **args)#,xlim=[0,50])
        plot(f"canJet{i}.pz", **args)#,yscale="log")


    for jetName in ["selJets", "othJets"]:
        for v in ["pt", "eta", "phi", "energy", "mass", "pz"]:
            plot(f"{jetName}.{v}",  **args)


    #
    #  Di-Jet Level
    #
    quad_jets = ["quadJet_min_dr","quadJet_selected"]
    di_jets = ["lead","subl","close","other"]

    for q in quad_jets:
        plot(f"{q}.xHH", **args)
        plot(f"{q}.dr", **args)
        for d in di_jets:
            plot(f"{q}.{d}.dphi", **args)
            plot(f"{q}.{d}.dr", **args)
            plot(f"{q}.{d}.eta", **args)
            plot(f"{q}.{d}.mass", **args)
            plot(f"{q}.{d}.phi", **args)
            plot(f"{q}.{d}.pt", **args)
            plot(f"{q}.{d}.pz", **args)

            #plot(f"{q}.{d}.dphi", **args)

    for v in ["v4j.pt", "v4j.phi", "v4j.eta", "v4j.mass"]:
        plot(v, **args)

    #
    # Top cand
    #
    for v in ["top_cand.xW", "xW",  "xbW", "top_cand.xWt", "top_cand.xWbW", "top_cand.xbW", "top_cand.rWbW",
              "top_cand.W.p.mass",  "top_cand.W.p.eta", "top_cand.W.p.pt",
              "top_cand.W.l.pt", "top_cand.W.j.pt", "top_cand.b.pt",
              "top_cand.W.l.phi", "top_cand.W.j.phi", "top_cand.b.phi",
              "top_cand.W.l.eta", "top_cand.W.j.eta", "top_cand.b.eta",
              "top_cand.t.mass",  "top_cand.t.eta", "top_cand.t.pt",
              ]:
        plot(v, **args)

    #
    #  Event Level
    #
    args["rebin"] = 1
    plot("selJets.n", **args, yscale="linear")
    plot("othJets.n", **args, yscale="linear")

    plot("hT", **args, yscale="linear")
    plot("hT_selected", **args, yscale="linear")

    args["rebin"] = 4
    plot("SvB_MA.ps_zh", **args, yscale="log")
    plot("SvB_MA.ps_zz", **args, yscale="log")
    plot("SvB_MA.ps_hh", **args, yscale="log")
    plot("SvB_MA.ptt", **args, yscale="log")
    plot("SvB_MA.tt_vs_mj", **args, yscale="log")
    plot("SvB_MA.ps_hh", **args, yscale="linear")
    #plot("SvB_MA.ps_hh", **args, yscale="linear")

    plot("SvB.ps_zh", **args, yscale="log")
    plot("SvB.ps_zz", **args, yscale="log")
    plot("SvB.ps_hh", **args, yscale="log")
    plot("SvB.tt_vs_mj", **args, yscale="log")
    plot("SvB.ptt", **args, yscale="log")

    plot("SvB.ps_hh", **args, yscale="linear")



def doPlots(doSignal=False, debug=False):

    norm = True


    args = {"norm": True,
            "doRatio": 1,
            "labels":["De-clustered","Nominal"],
            "norm": False,
            "region":"sum",
            "cut":"passPreSel",
            "doRatio":1,
            "rebin":1,
            "process":"data",
            "histtype":"step",
            }

    if doSignal:
        args["process"] = "HH4b"

    #for _cut in ["passPreSel", "pass0OthJets", "pass1OthJets"]:
    for _cut in ["passPreSel"]: #, "pass0OthJets", "pass1OthJets"]:
        args["cut"] = _cut
        print(f"plotting {_cut}")
        plotCut(args)





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
    doPlots(doSignal=args.signal, debug=args.debug)
