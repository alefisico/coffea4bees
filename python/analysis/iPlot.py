import os, time
#from coffea import hist, processor
import hist
import argparse
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np
from hist.intervals import ratio_uncertainty

#
# Move following to a config file?
#
axes = ["var","process","year","tag","region","cut"]
codeDicts = {}
codeDicts["tag"] = {"threeTag":3, "fourTag":4, 3:"threeTag", 4:"fourTag"}
codeDicts["region"]  = {"SR":2, "SB":1, 2:"SR", 1:"SB", 0:"other","other":0}


_phi = ( 1 + np.sqrt(5) ) / 2
_epsilon = 0.001

#
# TO Add
#     Variable Binning
#  - labels
#  output dir and file names

def ls(option="var", match=None):
    for k in axisLabels[option]:
        if match:
            if k.find(match) != -1: print(k)
        else:
            print(k)
        

def getHist(var='selJets.pt'):
    if var.find("*") != -1:
        ls(match=var.replace("*",""))
        return
    
    return hists['hists'][var]


def _draw_plot(hData, hBkg, **kwargs):
    r"""
    Takes options:
          "norm"   : bool
          "debug"  : bool
          "xlabel" : string
          "ylabel" : string
          "yscale" : 'log' | None
          "xscale" : 'log' | None
          "legend" : bool
          'ylim'   : [min, max]
          'xlim'   : [min, max]
    """
    
    norm = kwargs.get("norm",False)
    hData  .plot(density=norm, label="Data",     color="k", histtype="errorbar", markersize=7)

    
    # colors: https://xkcd.com/color/rgb/
    # hist options: https://mplhep.readthedocs.io/en/latest/api.html
    hBkg[0].plot(density=norm, color="k", histtype="step", yerr=False)
    hBkg[0].plot(density=norm, label="Multijet", color="xkcd:bright yellow", histtype="fill")


    #
    #  xlabel
    #
    if kwargs.get("xlabel",None): plt.xlabel(kwargs.get("xlabel"))
    plt.xlabel(plt.gca().get_xlabel(), fontsize=14, loc='right')

    
    #
    #  ylabel
    #
    if kwargs.get("ylabel",None): plt.ylabel(kwargs.get("ylabel"))
    if norm:   plt.ylabel(plt.gca().get_ylabel() + " (normalized)")
    font_properties = {'family': 'sans', 'weight': 'normal', 'size': 14, 'fontname':'Helvetica'}
    plt.ylabel(plt.gca().get_ylabel(), fontdict=font_properties, loc='top')

        
    if kwargs.get("yscale",None): plt.yscale(kwargs.get('yscale'))
    if kwargs.get("xscale",None): plt.xscale(kwargs.get('xscale'))
        

    if kwargs.get('legend',False):
        plt.legend(
            #loc='best',      # Position of the legend
            #fontsize='medium',      # Font size of the legend text
            frameon=False,           # Display frame around legend
            #framealpha=0.0,         # Opacity of the frame (1.0 is opaque)
            #edgecolor='black',      # Color of the legend frame
            #title='Trigonometric Functions',  # Title for the legend
            #title_fontsize='large',  # Font size of the legend title
            #bbox_to_anchor=(1, 1),   # Specify the position of the legend using bbox_to_anchor
            #borderaxespad=0.0       # Padding between the axes and the legend border
            reverse = True,
        )
        
    if kwargs.get('ylim',False):  plt.ylim(*kwargs.get('ylim'))
    if kwargs.get('xlim',False):  plt.xlim(*kwargs.get('xlim'))

    
    return
    



def _plot_ratio(hData, hBkg, **kwargs):
    r"""
    Takes options:
        "norm"              : bool (False),
        "ratio_line_value"  : number (1.0),
        "uncertainty_type"  : "poisson", "poisson-ratio", "efficiency" ("poisson")
        "rlim"              : [rmin, rmax] ([0,2])
    }
    """


    size = 7
    fig = plt.figure(figsize=(size, size/_phi*4/3))
    grid = fig.add_gridspec(2, 1, hspace=0.06, height_ratios=[3, 1], left=0.1, right=0.95, top=0.95, bottom=0.1 )

    main_ax    = fig.add_subplot(grid[0])

    _draw_plot(hData, hBkg, **kwargs)

    top_xlabel = plt.gca().get_xlabel()
    plt.xlabel("")
    
    subplot_ax = fig.add_subplot(grid[1], sharex=main_ax)
    plt.setp(main_ax.get_xticklabels(), visible=False)    

    numValues = hData.values()
    denValues = hBkg[0].values()
    denValues[denValues == 0] = _epsilon
    ratios = numValues / denValues
    
    if kwargs.get("norm",  False):
        numSF = np.sum(hData.values(), axis=0)
        denSF = np.sum(hBkg[0].values(), axis=0)
        ratios *= denSF/numSF
    
    # Set 0 and inf to nan to hide during plotting
    ratios[ratios == 0] = np.nan
    ratios[np.isinf(ratios)] = np.nan

    central_value_artist = subplot_ax.axhline(
        kwargs.get("ratio_line_value",1.0), color="black", linestyle="dashed", linewidth=1.0
    )
    
    # https://github.com/scikit-hep/hist/blob/main/src/hist/intervals.py
    ratio_uncert = ratio_uncertainty(
        num=numValues,
        denom=denValues,
        uncertainty_type=kwargs.get("uncertainty_type","poisson"),
    )
    
    x_values = hData.axes[0].centers

    subplot_ax.errorbar(
        x_values,
        ratios,
        yerr=ratio_uncert,
        color="black",
        marker="o",
        linestyle="none",
        markersize = 4,
    )

    #
    #  ylabel
    #
    plt.ylabel(kwargs.get("rlabel","Ratio"))
    font_properties = {'family': 'sans', 'weight': 'normal', 'size': 14, 'fontname':'Helvetica'}
    plt.ylabel(plt.gca().get_ylabel(), fontdict=font_properties, loc='center')


    plt.xlabel(kwargs.get("xlabel",top_xlabel),fontsize=14, loc='right')

    if kwargs.get('rlim',None):  plt.ylim(*kwargs.get('rlim'))
    
    return fig


def _plot(hData, hBkg, **kwargs):

    if kwargs.get("debug",False): print(f'\t kwargs = {kwargs}')

    size = 7
    fig = plt.figure(figsize=(size,size/_phi))

    fig.add_axes((0.1, 0.15, 0.85, 0.8))

    
    _draw_plot(hData, hBkg, **kwargs)

    ax = fig.gca()
    #ax.spines["top"]  .set_visible(False)
    #ax.spines["right"].set_visible(False)
    #ax.spines["left"] .set_visible(False)
    

    return fig
    


def plot(var='selJets.pt',year="2017",cut="passPreSel",tag="fourTag",region="SR", **kwargs):
    r"""
    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       year     : "2017",
       cut      : "passPreSel",
       tag      : "fourTag",
       region   : "SR",
    
       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    if kwargs.get("debug",False): print(f'kwargs = {kwargs}')

    
    if var.find("*") != -1:
        ls(match=var.replace("*",""))
        return 
    
    h = hists['hists'][var]
    
    #print(h)

    cutDict = {}
    for c in cutList:
        cutDict[c] = sum
    cutDict[cut] = True

    #
    #  Get Hists
    #
    rebin = kwargs.get("rebin",1)
    hData = h["mixeddata",  year,   hist.loc(codeDicts["tag"][tag]), hist.loc(codeDicts["region"][region]), cutDict[cutList[0]],  cutDict[cutList[1]], cutDict[cutList[2]],   hist.rebin(rebin)]
    hBkg  = h["mixeddata",  "UL17", hist.loc(codeDicts["tag"][tag]), hist.loc(codeDicts["region"][region]), cutDict[cutList[0]],  cutDict[cutList[1]], cutDict[cutList[2]], hist.rebin(rebin)]
    
    if kwargs.get("doRatio",False):
        fig = _plot_ratio(hData,[hBkg], **kwargs)
    else:
        fig = _plot(hData,[hBkg],**kwargs)

    fileName = "test.pdf"
    fig.savefig(fileName)
    plt.close()
    os.system("open "+fileName)
    
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', dest="inputFile", default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o','--outputFolder', dest="output_path", default='plots/', help='Folder for output folder. Default: plots/')
    args = parser.parse_args()

    if not os.path.exists(args.output_path): os.makedirs(args.output_path)
    with open(f'{args.inputFile}', 'rb') as hfile:
        hists = load(hfile)

        
        axisLabels = {}
        axisLabels["var"] = hists['hists'].keys()
        var1 = list(hists['hists'].keys())[0]

        cutList = []
        
        for a in hists["hists"][var1].axes:
            axisName = a.name
            if axisName == var1: continue

            if type(a) == hist.axis.Boolean:
                print(f"Adding cut\t{axisName}")
                cutList.append(axisName)
                continue

            if a.extent > 20: continue # HACK to skip the variable bins FIX
            axisLabels[axisName] = []
            print(axisName)
            for iBin in range(a.extent):
                if axisName in codeDicts:
                    value = codeDicts[axisName][a.value(iBin)]

                else:
                    value = a.value(iBin)
                    
                print(f"\t{value}")
                axisLabels[axisName].append(value)

                

