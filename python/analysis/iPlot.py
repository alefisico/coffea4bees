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

    


def plot(var='selJets.pt',year="2017",cut="passPreSel",tag="fourTag",region="SR", doratio=True, norm=False):

    if var.find("*") != -1:
        ls(match=var.replace("*",""))
        return 
    
    h = hists['hists'][var]
    
    #print(h)

    cutDict = {}
    for c in cutList:
        cutDict[c] = sum
    cutDict[cut] = True
    
    hZH = h["mixeddata",  year, hist.loc(codeDicts["tag"][tag]), hist.loc(codeDicts["region"][region]), cutDict[cutList[0]],  cutDict[cutList[1]], cutDict[cutList[2]], :]
    hHH = h["mixeddata",  "UL17", hist.loc(codeDicts["tag"][tag]), hist.loc(codeDicts["region"][region]), cutDict[cutList[0]],  cutDict[cutList[1]], cutDict[cutList[2]], :]

    if doratio:
        fig = plt.figure(figsize=(10, 8))
        main_ax_artists, ratio_ax_arists = hZH.plot_ratio(
            hHH,
            rp_ylabel=r'Four/Three',
            rp_num_label='FourTag',
            rp_denom_label='ThreeTag',
            rp_uncert_draw_type='line', # line or bar
        )
        axes = fig.get_axes()
        axes[1].set_ylim([0,2])
    else:
        fig = plt.figure(figsize=(8, 6))
        hZH.plot(density=norm)
        hHH.plot(density=norm)

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

                

