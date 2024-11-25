import sys
import os
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import parse_args, init_config, makePlot
import numpy as np
import yaml

def write_1D_pdf(output_file, bin_upper_edge, weights, n_spaces=0):
    spaces = " " * n_spaces
    output_file.write(f"{spaces}bin_upper_edge:  {bin_upper_edge.tolist()}\n")
    output_file.write(f"{spaces}weights:      {weights.tolist()}\n")


def doPlots(year):
    f, a = makePlot(cfg, var="hT_selected",region="sum",norm=0,doratio=1,rebin=1,yscale="linear",year=year,debug=True)
    bin_centers = a[1].lines[1].get_xydata()[:,0]
    bin_upper_edge = (bin_centers[:-1] + bin_centers[1:]) / 2
    weights     = a[1].lines[1].get_xydata()[:,1]
    weights[np.isnan(weights)] = 0

    with open(f"{args.outputFolder}/hT_weights_{year}.yml", 'w') as output_file:
        write_1D_pdf(output_file, bin_upper_edge, weights)



if __name__ == '__main__':

    args = parse_args()
    cfg = init_config(args)
    print(cfg)
    years = ["UL18", "UL17", "UL16_preVFP", "UL16_postVFP"]
    for y in years:
        doPlots(year=y)
