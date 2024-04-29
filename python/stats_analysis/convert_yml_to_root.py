import os, sys
import ROOT
import argparse
import logging
import yaml
import json
import numpy as np
ROOT.gROOT.SetBatch(True)


def json_to_TH1( coffea_hist, iname, rebin ):
    """docstring for hist_to_root"""

    edges     = coffea_hist['edges']
    centers   = coffea_hist['centers']
    values    = coffea_hist['values']
    variances = coffea_hist['variances']

    rHist = ROOT.TH1F(iname, iname, len(centers), edges[0], edges[-1])
    rHist.Sumw2()

    for ibin in range(1, len(centers) ):
        rHist.SetBinContent(ibin, values[ibin-1])
        rHist.SetBinError(ibin, ROOT.TMath.Sqrt(variances[ibin-1]))

    rHist.Rebin( rebin )

    return rHist

def create_root_file(file_to_convert, histos, output_dir):
    print( "in create_root_file")
    coffea_hists = yaml.safe_load(open(file_to_convert, 'r'))
    print( "leaded coffea_hists")

    root_hists = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print( "made dirs")
    output = output_dir + "/" + (file_to_convert.split("/")
                                 [-1].replace(".json", "")) + ".root"

    root_file = ROOT.TFile(output, 'recreate')

    for ih in coffea_hists.keys():
        for iprocess in coffea_hists[ih].keys():
            for iy in coffea_hists[ih][iprocess].keys():
                for itag in coffea_hists[ih][iprocess][iy].keys():
                    for iregion in coffea_hists[ih][iprocess][iy][itag].keys():
                        this_hist = json_to_TH1(
                            coffea_hists[ih][iprocess][iy][itag][iregion],
                            ih.replace(".", "_") + "_" + iprocess + "_" + iy + "_" + itag + "_" + iregion,
                            1)
                        print( 'Converting hist', ih, ih.replace(".", "_") + "_" + iprocess + "_" + iy + "_" + itag + "_" + iregion)
                        this_hist.Write()

    root_file.Close()
    logging.info("\n File " + output + " created.")


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Convert json hist to root TH1F', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', dest="output_dir",
                        default="./datacards/", help='Output directory.')
    parser.add_argument('--histos', dest="histos", nargs="+",
                        default=[ ], help='List of histograms to convert')
    parser.add_argument('--classifier', dest="classifier", nargs="+",
                        default=["SvB_MA", "SvB"], help='Classifier to make histograms.')
    parser.add_argument('-f', '--file', dest='file_to_convert',
                        default="histos/histAll.json", help="File with coffea hists")
    parser.add_argument('-s', '--syst_file', dest='systematics_file',
                        default='', help="File contain systematic variations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("\nRunning with these parameters: ")
    logging.info(args)

    logging.info("Creating root files from json")
    create_root_file(args.file_to_convert, args.histos, args.output_dir)
