import argparse
import logging
from coffea.util import load, save

if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Merge several coffea files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', dest="output_file",
                        default="hists.coffea", help='Output file.')
    parser.add_argument('-f', '--files', nargs='+', dest='files_to_merge', default=[], help="List of files to merge")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")

    output = {}

    output = load(args.files_to_merge[0])
    for ifile in args.files_to_merge[1:]:
        logging.info(f'Merging {ifile}')
        iout = load(ifile)
        for ikey in output.keys():
            if 'hists' in ikey:
                for ihist in output['hists'].keys():
                    try:
                        output['hists'][ihist] += iout['hists'][ihist]
                    except KeyError:
                        pass
            else:
                output[ikey] = output[ikey] | iout[ikey]


    hfile = f'{args.output_file}'
    logging.info(f'\nSaving file {hfile}')
    save(output, hfile)
