import yaml
import sys
import os
sys.path.insert(0, os.getcwd())
from base_class.plots.helpers_make_plot import _plot_from_dict
import base_class.plots.helpers as plot_helpers
import argparse

parser = argparse.ArgumentParser(description='plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_yaml_files', nargs='+', help='Input yaml file')

parser.add_argument('-o', '--outputFolder', default="./",
                    help='Folder for output folder. Default: ./')

args = parser.parse_args()


for input_yaml_file in args.input_yaml_files:
    with open(input_yaml_file, "r") as yfile:
        loaded_data = yaml.safe_load(yfile)


    loaded_data["kwargs"]["outputFolder"] = args.outputFolder


    VARIANCE_RATIO_GRID_CONFIG = {
        'hspace': 0.04,
        'height_ratios': [1, 1],
        'left': 0.1,
        'right': 0.95,
        'top': 0.95,
        'bottom': 0.1
    }

    # No need to rewrite the yaml
    loaded_data["kwargs"]["write_yaml"] = False
    loaded_data["kwargs"]["ratio_grid_config"] = VARIANCE_RATIO_GRID_CONFIG

    #         make_plot_from_dict(loaded_data)
    fig, main_ax, ratio_ax = _plot_from_dict(loaded_data, **loaded_data["kwargs"])
    ax = (main_ax, ratio_ax)

    kwargs = loaded_data["kwargs"]

    #rMin = kwargs.get("rlim", [0,2])[0]
    #rMax = kwargs.get("rlim", [0,2])[1]

    for i in range(15):
        ratio_ax.axvline(x=i,  ymin=0, ymax=1,       color='black', linewidth=1.25)
        main_ax .axvline(x=i,  ymin=0, ymax=0.6,    color='black', linewidth=1.25)


    output_path = [
        kwargs.get("outputFolder"),
    ]

    # Determine file name
    file_name = loaded_data.get("file_name", loaded_data["var"])
    if kwargs.get("yscale", None) == "log":
        file_name += "_logy"

    # Save plot
    plot_helpers.savefig(fig, file_name, *output_path)
