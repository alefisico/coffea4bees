# source /cvmfs/sft.cern.ch/lcg/views/LCG_102rc1/x86_64-centos7-gcc11-opt/setup.sh
# source /cvmfs/sft.cern.ch/lcg/nightlies/dev4/Wed/coffea/0.7.13/x86_64-centos7-gcc10-opt/coffea-env.sh
import pickle, os, time
#from coffea import hist, processor
import hist
import argparse
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coffea.util import load

#
# Move following to a config file
#
axes = ["var","process","year","tag","region","cut"]
codeDicts = {}
codeDicts["tag"] = {"threeTag":3, "fourTag":4, 3:"threeTag", 4:"fourTag"}
codeDicts["region"]  = {"SR":2, "SB":1, 2:"SR", 1:"SB", 0:"other","other":0}


#
# TO Add
#  - Rebin
#     Variable Binning
#  - Normalized
#  - x Max /min
#  - labels
#  - ratio 



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




def _plot_ratio(hData, hBkg, *kwargs):
    print(kwargs)
    fig = plt.figure(figsize=(10, 8))
    main_ax_artists, ratio_ax_arists = hData.plot_ratio(
        hBkg[0],
        rp_ylabel=r'Four/Three',
        rp_num_label='FourTag',
        rp_denom_label='ThreeTag',
        rp_uncert_draw_type='line', # line or bar
    )
    axes = fig.get_axes()
    axes[1].set_ylim([0,2])
    return fig


def _plot(hData, hBkg, **kwargs):
    opts = {"norm"   : False,
            "debug"  : False,
            "xlabel" : None,
            "ylabel" : None,
            "yscale" : None,
            "xscale" : None,
            "legend" : False,
            'ylim'   : None,
            'xlim'   : None,
               }

    for key, value in opts.items():
        opts[key] = kwargs.get(key,  value)

    if opts["debug"]: print(f'\t opts = {opts}')
    print(f'\t opts = {opts}')
    
    fig = plt.figure(figsize=(8, 6))
    hData  .plot(density=opts["norm"], label="Data",     color="k", histtype="errorbar", markersize=7)

    # colors: https://xkcd.com/color/rgb/
    # hist options: https://mplhep.readthedocs.io/en/latest/api.html
    hBkg[0].plot(density=opts["norm"], color="k", histtype="step", yerr=False)
    hBkg[0].plot(density=opts["norm"], label="Multijet", color="xkcd:bright yellow", histtype="fill")


    #
    #  xlabel
    #
    if opts["xlabel"]: plt.xlabel(opts["xlabel"])
    plt.xlabel(plt.gca().get_xlabel(), fontsize=14, loc='right')


    
    #
    #  ylabel
    #
    if opts["ylabel"]: plt.ylabel(opts["ylabel"])
    if opts["norm"]:         plt.ylabel(plt.gca().get_ylabel() + " (normalized)")
    font_properties = {'family': 'sans', 'weight': 'normal', 'size': 14, 'fontname':'Helvetica'}
    plt.ylabel(plt.gca().get_ylabel(), fontdict=font_properties, loc='top')

        
    if opts['yscale']: plt.yscale(opts['yscale'])
    if opts['xscale']: plt.xscale(opts['xscale'])
        

    if opts['legend']:
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
        #plt.legend()
        
    if opts['ylim']:  plt.ylim(opts['ylim'][0],opts['ylim'][1])
    if opts['xlim']:  plt.xlim(opts['xlim'][0],opts['xlim'][1])        
        
        

    return fig
    


def plot(var='selJets.pt',year="2017",cut="passPreSel",tag="fourTag",region="SR", **kwargs):
    debug   = kwargs.get('debug', False)
    doratio = kwargs.get('doratio', False)
    if debug: print(f'kwargs = {kwargs}')

    
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
    hData = h["mixeddata",  year, hist.loc(codeDicts["tag"][tag]), hist.loc(codeDicts["region"][region]), cutDict[cutList[0]],  cutDict[cutList[1]], cutDict[cutList[2]], :]
    hBkg  = h["mixeddata",  "UL17", hist.loc(codeDicts["tag"][tag]), hist.loc(codeDicts["region"][region]), cutDict[cutList[0]],  cutDict[cutList[1]], cutDict[cutList[2]], :]

#    if doratio:
#        plot_ratio(hData,[hBkg],rp_ylabel=r'Four/Three',
#                   rp_num_label='FourTag',
#                   rp_denom_label='ThreeTag',
#                   rp_uncert_draw_type='line', # line or bar
#                   )
#            
    
    if doratio:
        fig = _plot_ratio(hData,[hBkg])
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
        #hists = pickle.load(hfile)
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

                

