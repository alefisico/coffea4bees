import yaml
import sys
import os
sys.path.insert(0, os.getcwd())
from base_class.plots.plots import make_plot_from_dict, make_plot_2d_from_dict
import argparse

parser = argparse.ArgumentParser(description='plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_yaml', help='Input yaml file')

parser.add_argument('-o', '--outputFolder', default="./",
                    help='Folder for output folder. Default: ./')

parser.add_argument('--do_2d', action="store_true")

args = parser.parse_args()


with open(args.input_yaml, "r") as yfile:
    loaded_data = yaml.safe_load(yfile)


loaded_data["kwargs"]["outputFolder"] = args.outputFolder

# No need to rewrite the yaml
loaded_data["kwargs"]["write_yaml"] = False

if args.do_2d:
    make_plot_2d_from_dict(loaded_data)
else:
    make_plot_from_dict(loaded_data)
