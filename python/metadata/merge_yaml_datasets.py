import yaml
import argparse


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--main_file', dest="main_file", default="datasets_HH4b.yml", help='Main datasets file.')
    parser.add_argument('-o', '--output_file', dest="output_file", default="datasets_HH4b_merged.yml", help='Output file.')
    parser.add_argument('-f', '--files_to_add', nargs='+', dest="files_to_add", default=["datasets_HH4b.yml"], help='Files to add.')
    args = parser.parse_args()

    main_file = yaml.safe_load(open(args.main_file, 'r'))

    for ifile in args.files_to_add:

        tmp_file = yaml.safe_load(open(ifile, 'r'))

        for ikey in tmp_file.keys():
            tmp_split = ('_UL' if 'UL' in ikey else '_20')
            dataset = ikey.split( tmp_split )[0]
            year = tmp_split.split('_')[1] + '_'.join(ikey.split(tmp_split)[1:])
            if dataset in main_file['datasets']:
                main_file['datasets'][dataset][year]['picoAOD'] = tmp_file[ikey]

    yaml.dump(main_file, open(args.output_file, 'w'), default_flow_style=False)
