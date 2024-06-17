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


def write_1D_pdf(output_file, varName, bin_centers, probs):
    output_file.write(f"{varName}:\n")
    output_file.write(f"    bin_centers:  {bin_centers.tolist()}\n")
    output_file.write(f"    probs:  {probs.tolist()}\n")
    

def doPlots(debug=False):

    #
    #  Synthetic datasets
    #
    varNames = ["gbbs.thetaA", "gbbs.mA", "gbbs.mB", "gbbs.zA", "gbbs.decay_phi"]
    #varNames = ["gbbs.thetaA"]

    output_file_name = args.outputFolder+"/clustering_pdfs.yml"
    with open(f'{output_file_name}', 'w') as output_file:

    
        for _v in varNames:
    
            fig, ax = plot(_v, region="SR", cut="passPreSel",doRatio=0,rebin=1,process="data")
            bin_centers = ax.get_lines()[0].get_xdata()
            counts      = ax.get_lines()[0].get_ydata()    

            probs = counts / counts.sum()
            
            write_1D_pdf(output_file, _v, bin_centers, probs)


    with open(output_file_name, 'r') as input_file:

        input_pdfs = yaml.safe_load(input_file)
        
        for _v in varNames:

            probs   = np.array(input_pdfs[_v]["probs"], dtype=float)
            centers = np.array(input_pdfs[_v]["bin_centers"], dtype=float)
            
            num_samples = 10000
            samples = np.random.choice(centers, size=num_samples, p=probs)

            nBins = len(centers)
            bin_half_width = 0.5*(centers[1]  - centers[0])
            xMin  = centers[0]  - bin_half_width
            xMax  = centers[-1] + bin_half_width
            
            sample_hist = hist.Hist.new.Reg(nBins, xMin, xMax).Double()
            sample_hist.fill(samples)
        
            sample_pdf  = hist.Hist.new.Reg(nBins, xMin, xMax).Double()
            sample_pdf[...] = probs * num_samples
        
            sample_hist.plot(label="samples")
            sample_pdf.plot(label="pdf")    
            plt.xlabel(_v)
            plt.legend()
            plt.savefig(args.outputFolder+f"/test_sample_{_v}.pdf")


            #plt.legend()
            plt.close()
    
    


    #breakpoint()
    
        
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
