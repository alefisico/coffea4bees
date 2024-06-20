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


def plot2d(process, **kwargs):
    fig, ax = make2DPlot(cfg, process, outputFolder= args.outputFolder, **kwargs)
    plt.close()
    return fig, ax


def write_1D_pdf(output_file, varName, bin_centers, probs):
    output_file.write(f"    {varName}:\n")
    output_file.write(f"        bin_centers:  {bin_centers.tolist()}\n")
    output_file.write(f"        probs:  {probs.tolist()}\n")


def write_2D_pdf(output_file, varName, xcenters_flat, ycenters_flat, probabilities_flat):
    output_file.write(f"    {varName}:\n")
    output_file.write(f"        xcenters_flat:  {xcenters_flat.tolist()}\n")
    output_file.write(f"        ycenters_flat:  {ycenters_flat.tolist()}\n")
    output_file.write(f"        probabilities_flat:  {probabilities_flat.tolist()}\n")




def doPlots(debug=False):

    #
    #  Synthetic datasets
    #

    gbb_hist_name = {}
    #gbb_hist_name["thetaA"]    = ("gbbs.thetaA",    1)  # (hist_name, rebin)
    #gbb_hist_name["mA"]        = ("gbbs.mA",        1)
    #gbb_hist_name["mB"]        = ("gbbs.mB",        1)
    gbb_hist_name["rhoA"]      = ("gbbs.rhoA",      1)
    gbb_hist_name["rhoB"]      = ("gbbs.rhoB",      1)
    #gbb_hist_name["zA"]        = ("gbbs.zA",        1)
    gbb_hist_name["decay_phi"] = ("gbbs.decay_phi", 4)
    gbb_hist_name["zA_vs_thetaA"]        = ("gbbs.zA_vs_thetaA",        1)

    bstar_hist_name = {}
    #bstar_hist_name["thetaA"]    = ("bstars.thetaA_l" ,  1)
    # bstar_hist_name["mA"]        = ("bstars.mA"       ,  1)
    # bstar_hist_name["mB"]        = ("bstars.mB"       ,  1)
    bstar_hist_name["rhoA"]      = ("bstars.rhoA"     ,  1)
    bstar_hist_name["rhoB"]      = ("bstars.rhoB"     ,  1)
    #bstar_hist_name["zA"]        = ("bstars.zA_l"     ,  1)
    bstar_hist_name["decay_phi"] = ("bstars.decay_phi",  4)
    bstar_hist_name["zA_vs_thetaA"]        = ("gbbs.zA_vs_thetaA",        1)

    splitting_hist_name = {}
    splitting_hist_name["gbbs"]   = gbb_hist_name
    splitting_hist_name["bstars"] = bstar_hist_name

    varNames   = list(gbb_hist_name.keys())
    splittings = list(splitting_hist_name.keys())

    output_file_name = args.outputFolder+"/clustering_pdfs.yml"
    with open(f'{output_file_name}', 'w') as output_file:

        output_file.write("varNames:\n")
        output_file.write(f"    {varNames}\n\n")

        output_file.write("splittings:\n")
        output_file.write(f"    {splittings}\n\n")

        #
        #  Write the 1D PDFs
        #
        for _s in splittings:
            output_file.write(f"\n{_s}:\n")

            for _v in varNames:

                if _v.find("_vs_") == -1:
                    is_1d_hist = True
                else:
                    is_1d_hist = False

                if is_1d_hist:
                    _hist_name, _rebin = splitting_hist_name[_s][_v]

                    fig, ax = plot(_hist_name, region="SR", cut="passPreSel", doRatio=0, rebin=_rebin, process="data")
                    bin_centers = ax.get_lines()[0].get_xdata()
                    counts      = ax.get_lines()[0].get_ydata()

                    probs = counts / counts.sum()

                    write_1D_pdf(output_file, _v, bin_centers, probs)

                else:

                    hist_to_plot = cfg.hists[0]["hists"][f"{_s}.{_v}"]
                    _hist = hist_to_plot[{"process":"data", "year":sum, "tag":1,"region":0,"passPreSel":True}]

                    counts = _hist.view(flow=False)

                    xedges = _hist.axes[0].edges
                    yedges = _hist.axes[1].edges
                    probabilities = counts.value / counts.value.sum()

                    xcenters = (xedges[:-1] + xedges[1:]) / 2
                    ycenters = (yedges[:-1] + yedges[1:]) / 2

                    xcenters_flat = np.repeat(xcenters, len(ycenters))
                    ycenters_flat = np.tile(ycenters, len(xcenters))
                    probabilities_flat = probabilities.flatten()

                    write_2D_pdf(output_file, _v, xcenters_flat, ycenters_flat, probabilities_flat)



    with open(output_file_name, 'r') as input_file:

        input_pdfs = yaml.safe_load(input_file)

        for _s in splittings:

            for _v in varNames:

                if _v.find("_vs_") == -1:
                    is_1d_hist = True
                else:
                    is_1d_hist = False

                if is_1d_hist:
                    probs   = np.array(input_pdfs[_s][_v]["probs"],       dtype=float)
                    centers = np.array(input_pdfs[_s][_v]["bin_centers"], dtype=float)

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
                    plt.savefig(args.outputFolder+f"/test_sampling_{_s}_{_v}.pdf")

                    plt.close()

                else:
                    #
                    # 2D Vars
                    #
                    _v = "zA_vs_thetaA"
                    probabilities_flat   = np.array(input_pdfs[_s][_v]["probabilities_flat"],       dtype=float)
                    xcenters_flat        = np.array(input_pdfs[_s][_v]["xcenters_flat"], dtype=float)
                    ycenters_flat        = np.array(input_pdfs[_s][_v]["ycenters_flat"], dtype=float)

                    num_samples = 10000

                    # Draw samples
                    sampled_indices = np.random.choice(len(probabilities_flat), size=num_samples, p=probabilities_flat)
                    sampled_x = xcenters_flat[sampled_indices]
                    sampled_y = ycenters_flat[sampled_indices]

                    # Plot the original 2D histogram
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)

                    probs2d = probabilities_flat.reshape(50,50)
                    plt.imshow(probs2d.transpose(), cmap='Blues', origin='lower')

                    plt.title('Original Data Histogram')

                    # Plot the sampled data
                    plt.subplot(1, 2, 2)
                    plt.hist2d(sampled_x, sampled_y, bins=[xedges, yedges], cmap='Blues')
                    plt.title('Sampled Data Histogram')
                    plt.xlabel('X')
                    plt.ylabel('Y')


                    plt.savefig(args.outputFolder+f"/test_sampling_{_s}_{_v}.pdf")

                    plt.close()

                    #plt.show()




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
