import os, sys
import argparse
import logging
import yaml
import numpy as np
from coffea.util import load
from convert_hist_to_yaml import hist_to_yml



if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert yml hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hist_key', nargs="+",
                        default=['ps_zz',      'ps_zh',      'ps_hh', 
                                 'ps_zz_fine', 'ps_zh_fine', 'ps_hh_fine'],
                        help='List of histograms to convert')
                             
    parser.add_argument('-o', '--output', dest="output",
                        default=None, help='Output file and directory.')

    parser.add_argument('-i', '--input_file', dest='input_file',
                        default="../analysis/hists/histAll.coffea", help="File with coffea hists")

    parser.add_argument("--debug", action="store_true")
    #parser.add_argument("--signal", action="store_true")
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


    #
    # Collect the histogram names
    #
    hists_to_save = []
    for k in coffea_hists.keys():
        for _hist_key in args.hist_key:
            if not k.find(_hist_key) == -1:
                hists_to_save.append(k)
        
    print(hists_to_save)

    # Remove duplicates
    hists_to_save = set(hists_to_save)

    #
    #  Criteria to save
    #
    save_dict = {}
    for sub_sample in range(15):
        save_dict[f"mix_v{sub_sample}"] = [('fourTag','SR')]

    save_dict["data_3b_for_mixed"]          = [('threeTag','SR')]
    save_dict["TTTo2L2Nu_for_mixed"]        = [('fourTag','SR')]
    save_dict["TTToSemiLeptonic_for_mixed"] = [('fourTag','SR')]
    save_dict["TTToHadronic_for_mixed"]     = [('fourTag','SR')]
    save_dict["ZZ4b"]     = [('fourTag','SR')]
    save_dict["ZH4b"]     = [('fourTag','SR')]
    save_dict["HH4b"]     = [('fourTag','SR')]


    yml_dict = {}
    for ih in hists_to_save:
        yml_dict[ih] = {}
        for iprocess in coffea_hists[ih].axes[0]:
            if iprocess not in save_dict: 
                print(f"skipping process {iprocess}")
                continue

            yml_dict[ih][iprocess] = {}

            for iy in coffea_hists[ih].axes[1]:
                yml_dict[ih][iprocess][iy] = {}

                for itag in range(len(coffea_hists[ih].axes[3])):
                    yml_dict[ih][iprocess][iy][codes['tag'][itag]] = {}

                    
                    for iregion in range(len(coffea_hists[ih].axes[4])):
                        
                        tag_region_pair = (codes['tag'][itag], codes['region'][iregion])

                        if tag_region_pair not in save_dict[iprocess]:
                            if args.debug: 
                                print(f"skipping {iprocess} {tag_region_pair}")
                            continue
                        
                        this_hist = {
                            'process' : iprocess,
                            'year' : iy,
                            'tag' : itag,
                            'region' : iregion,
                            'passPreSel' : True,
                            'passSvB' : sum,
                            'failSvB' : sum
                        }
                        logging.info(f"Converting hist {ih} {this_hist}")
                        yml_dict[ih][iprocess][iy][codes['tag'][itag]][codes['region'][iregion]] = hist_to_yml( coffea_hists[ih][this_hist] )

    if args.output is None:
        output = args.input_file.replace(".coffea",".yml")
    else:
        output = args.output

    logging.info(f"Saving histos in yml format in {output}")
    output_dir = '/'.join( output.split('/')[:-1] )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    yaml.dump(yml_dict, open(f'{output}', 'w'), default_flow_style=False )
