import os, sys
import argparse
import logging
import yaml
import numpy as np
from coffea.util import load


def hist_to_yml( coffea_hist ):
    """docstring for hist_to_root"""

    yhist = {
        'edges' : coffea_hist.axes[0].edges.tolist(),
        'centers' : coffea_hist.axes[0].centers.tolist(),
        'values' : coffea_hist.values(),
        'variances' : coffea_hist.variances(),
    }

    ### in case of negative values
    yhist['values'] = np.where( yhist['values']<0, 0, yhist['values'] ).tolist()
    yhist['variances'] = np.where( yhist['variances']<0, 0, yhist['variances'] ).tolist()

    return yhist


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert yml hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--histos', dest="histos", nargs="+",
                        default=['SvB.ps_zz', 'SvB.ps_zh', 'SvB.ps_hh', 'SvB_MA.ps_zz', 'SvB_MA.ps_zh',
                                 'SvB_MA.ps_hh' ], help='List of histograms to convert')
    parser.add_argument('-o', '--output', dest="output",
                        default="./histos/histAll.yml", help='Output file and directory.')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default="../analysis/hists/histAll.coffea", help="File with coffea hists")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")

    codes = {
        'region' : {
            2 : 'other',
            1 : 'SB',
            0 : 'SR'
        },
        'tag' : {
            0 : 'threeTag',
            1 : 'fourTag',
            2 : 'other'
        }
    }

    coffea_hists = load(args.input_file)["hists"]

    yml_dict = {}
    for ih in args.histos:
        yml_dict[ih] = {}
        for iprocess in coffea_hists[ih].axes[0]:
            yml_dict[ih][iprocess] = {}
            for iy in coffea_hists[ih].axes[1]:
                yml_dict[ih][iprocess][iy] = {}
                for isys in coffea_hists[ih].axes[2]:
                    yml_dict[ih][iprocess][iy][isys] = {}
                    for itag in range(len(coffea_hists[ih].axes[3])):
                        yml_dict[ih][iprocess][iy][isys][codes['tag'][itag]] = {}
                        for iregion in range(len(coffea_hists[ih].axes[4])):
                            this_hist = {
                                'process' : iprocess,
                                'year' : iy,
                                'variation' : isys,
                                'tag' : itag,
                                'region' : iregion,
                                'passPreSel' : True,
                                'passSvB' : sum,
                                'failSvB' : sum
                            }
                            logging.info(f"Converting hist {ih} {this_hist}")
                            yml_dict[ih][iprocess][iy][isys][codes['tag'][itag]][codes['region'][iregion]] = hist_to_yml( coffea_hists[ih][this_hist] )

    logging.info(f"Saving histos in yml format in {args.output}")
    output_dir = '/'.join( args.output.split('/')[:-1] )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    yaml.dump(yml_dict, open(f'{args.output}', 'w'), default_flow_style=False )
