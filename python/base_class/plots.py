import os
import hist
from hist.intervals import ratio_uncertainty
import matplotlib.pyplot as plt
import numpy as np

_phi = ( 1 + np.sqrt(5) ) / 2
_epsilon = 0.001


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
    
    if kwargs.get("debug",False): print(f'\t in _draw_plot ... kwargs = {kwargs}')
    
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



def _plot(hData, hBkg, **kwargs):

    if kwargs.get("debug",False): print(f'\t in plot ... kwargs = {kwargs}')

    size = 7
    fig = plt.figure(figsize=(size,size/_phi))

    fig.add_axes((0.1, 0.15, 0.85, 0.8))
    
    _draw_plot(hData, hBkg, **kwargs)

    #ax = fig.gca()
    #ax.spines["top"]  .set_visible(False)
    #ax.spines["right"].set_visible(False)
    #ax.spines["left"] .set_visible(False)
    

    return fig



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

    plt.ylim(*kwargs.get('rlim',[0,2]))
    
    return fig



def makePlot(hists, cutList, plotConfig, var='selJets.pt',cut="passPreSel",region="SR", **kwargs):
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

    
    h = hists['hists'][var]
    varName = hists['hists'][var].axes[-1].name

    cutDict = {}
    for c in cutList:
        cutDict[c] = sum
    cutDict[cut] = True

    proc1 = plotConfig["plot1"]["process"]
    year1 = sum if plotConfig["plot1"]["year"] == "RunII" else plotConfig["plot1"]["year"]
    tag1  = plotConfig["codes"]["tag"][plotConfig["plot1"]["tag"]]

    proc2 = plotConfig["plot2"]["process"]
    year2 = sum if plotConfig["plot2"]["year"] == "RunII" else plotConfig["plot2"]["year"]
    tag2  = plotConfig["codes"]["tag"][plotConfig["plot2"]["tag"]]

    #
    #  Get Hists
    #
    rebin = kwargs.get("rebin",1)
    hDataDict = {"process":proc1,  "year":year1,   "tag":hist.loc(tag1), "region":hist.loc(plotConfig["codes"]["region"][region]), varName:hist.rebin(rebin)}
    hBkgDict  = {"process":proc2,  "year":year2,   "tag":hist.loc(tag2), "region":hist.loc(plotConfig["codes"]["region"][region]), varName:hist.rebin(rebin)}
    for c in cutList:
        hDataDict = hDataDict | {c:cutDict[c]}
        hBkgDict  = hBkgDict  | {c:cutDict[c]}

    hData = h[hDataDict]
    hBkg  = h[hBkgDict]
              
    
    if kwargs.get("doRatio",False):
        fig = _plot_ratio(hData,[hBkg], **kwargs)
    else:
        fig = _plot(hData,[hBkg],**kwargs)

    if kwargs.get("outputFolder",None):
        yearName = plotConfig["plot1"]["year"]
        tagName  = plotConfig["plot1"]["tag"]
        outputPath = kwargs.get("outputFolder")+"/"+"/".join([yearName,cut,tagName,region])
        if not os.path.exists(outputPath): os.makedirs(outputPath)
        fig.savefig(outputPath+"/"+var+".pdf")        
        
    return fig
