import os
import sys
import yaml
import hist
import matplotlib.pyplot as plt
from hist.intervals import ratio_uncertainty
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import makePlot, make2DPlot, load_config, load_hists, read_axes_and_cuts, parse_args, print_cfg
import base_class.plots.iPlot_config as cfg

#
# TO Add
#     Variable Binning


def ls(option="var", match=None):
    for k in cfg.axisLabels[option]:
        if match:
            if k.find(match) != -1:
                print(k)
        else:
            print(k)

def info():
    print_cfg(cfg)

def examples():
    print("examples:\n\n")
    print(
        '# Nominal plot of data and background in the a region passing a cut \n'
        'plot("v4j.mass", region="SR", cut="passPreSel")\n\n'

        '# Can get a print out of the varibales\n'
        'ls()'
        'plot("*", region="SR", cut="passPreSel")\n'
        'plot("v4j*", region="SR", cut="passPreSel")\n\n'

        '# Can add ratio\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1)\n\n'

        '# Can rebin\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4)\n\n'

        '# Can normalize\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1)\n\n'

        '# Can set logy\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, yscale="log")\n\n'

        '# Can set ranges\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, rlim=[0.5,1.5])\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, xlim=[0,1000])\n'
        'plot("v4j.mass", region="SR", cut="passPreSel", doRatio=1, rebin=4, norm=1, ylim=[0,0.01])\n\n'

        '# Can overlay different regions \n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="data", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="HH4b", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="Multijet", doRatio=1, rebin=4)\n'
        'plot("v4j.mass", region=["SR","SB"], cut="passPreSel", process="TTToHadronic", doRatio=1, rebin=4)\n\n'

        '# Can overlay different cuts \n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="data", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="HH4b", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="Multijet", doRatio=1, rebin=4, norm=1)\n'
        'plot("v4j.mass", region="SR", cut=["passPreSel","passSvB","failSvB"], process="TTToHadronic", doRatio=1, rebin=4, norm=1)\n\n'

        '# Can overlay different variables \n'
        'plot(["canJet0.pt","canJet1.pt"], region="SR",cut="passPreSel",doRatio=1,process="Multijet")\n'
        'plot(["canJet0.pt","canJet1.pt","canJet2.pt","canJet3.pt"], region="SR", cut="passPreSel",doRatio=1,process="Multijet")\n\n'

        '# Can plot a single process  \n'
        'plot("v4j.mass", region="SR", cut="passPreSel",process="data")\n\n'

        '# Can overlay processes  \n'
        'plot("v4j.mass", region="SR", cut="passPreSel",norm=1,process=["data","TTTo2L2Nu","HH4b","Multijet"],doRatio=1)\n\n'

        '# Plot 2d hists \n'
        'plot2d("quadJet_min_dr.close_vs_other_m",process="Multijet",region="SR",cut="failSvB")\n'
        'plot2d("quadJet_min_dr.close_vs_other_m",process="Multijet",region="SR",cut="failSvB",full=True)\n\n'

        '# Unsup4b plots with SB and SRSB as composite regions \n'
        'plot("v4j.mass", region="SRSB", cut="passPreSel") \n'
        'plot2d("quadJet_selected.lead_vs_subl_m",process="data3b",region="SRSB") \n'
        'plot("leadStM_selected", region="SB", cut="passPreSel", process = ["data3b","mixeddata"]) \n'
        'plot("v4j.mass", region=["SR", "SB"], cut="passPreSel", process = "data3b") \n\n'
    )


def plot(var='selJets.pt', *, cut="passPreSel", region="SR", **kwargs):
    r"""
    Plot

    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       cut      : "passPreSel",
       region   : "SR",

       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """
    
    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')
    
    if type(var) is not list and var.find("*") != -1:
        ls(match=var.replace("*", ""))
        return
    
    if len(cfg.hists) > 1:
        fig, ax = makePlot(cfg, var=var, cut=cut, region=region,
                           outputFolder=cfg.outputFolder, fileLabels=cfg.fileLabels, **kwargs)
    else:
        fig, ax = makePlot(cfg, var=var, cut=cut, region=region,
                           outputFolder=cfg.outputFolder, **kwargs)
    

    fileName = "test.pdf"
    fig.savefig(fileName)
    plt.close()
    os.system("open "+fileName)

    if kwargs.get("debug", False):
        return fig, ax


def plot2d(var='quadJet_selected.lead_vs_subl_m', process="HH4b",
           *, cut="passPreSel", region="SR", **kwargs):
    r"""
    Plot 2d

    Call with:
       plot2d("quadJet_selected.lead_vs_subl_m",process="data",region="SB",cut="passPreSel",tag="threeTag")
       plot2d("quadJet_selected.lead_vs_subl_m",process="HH4b",region="SR",cut="passPreSel",tag="threeTag")


    Takes Options:

       debug    : False,
       var      : 'quadJet_selected.lead_vs_subl_m',
       process  : 'HH4b',
       cut      : "passPreSel",
       region   : "SR",

       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    if kwargs.get("debug", False):
        print(f'kwargs = {kwargs}')

    if var.find("*") != -1:
        ls(match=var.replace("*", ""))
        return

    fig, ax = make2DPlot(cfg, process, var=var, cut=cut,
                         region=region, outputFolder=cfg.outputFolder, **kwargs)

    fileName = "test.pdf"
    fig.savefig(fileName)
    plt.close()
    os.system("open "+fileName)

    if kwargs.get("debug", False):
        return fig, ax



if __name__ == '__main__':
    args = parse_args()
    cfg.plotConfig = load_config(args.metadata)
    cfg.outputFolder = args.outputFolder

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
    print_cfg(cfg)

