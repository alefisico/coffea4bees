import logging
from argparse import ArgumentParser

import cloudpickle
import fsspec
import pandas as pd
from base_class.system.eos import EOS
from base_class.utils.argparser import DefaultFormatter
from rich.logging import RichHandler

from ..ikappa._sanity import group_by_categories, group_to_str


def group_hists(in_path: str, out_path: str):
    with fsspec.open(in_path, "rb", compression="lz4") as f:
        data = cloudpickle.load(f)

    categories, groups = group_by_categories(data["hists"], data["categories"])
    if failed := groups.pop(None, None):
        logging.warning(f"The following histograms have missing categories: {failed}")
    axes = pd.DataFrame(groups.keys(), columns=categories)
    axes = axes.loc[:, axes.nunique(axis=0) > 1]
    output = EOS(out_path)
    for i, group in enumerate(groups.values()):
        g = i + 1
        file = output.parent / f"{output.stem}_{g}.coffea"
        axis = axes.iloc[i]
        logging.info(f"Group {g}: {file}")
        for k, v in axis.items():
            logging.info(f"{k}: {group_to_str(v)}")
        logging.info("histograms: " + ", ".join(group))
        data = {
            "hists": {k: data["hists"][k] for k in group},
            "categories": data["categories"],
        }
        with fsspec.open(file, "wb", compression="lz4") as f:
            cloudpickle.dump(data, f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=False, show_path=False, markup=True)],
    )
    parser = ArgumentParser(formatter_class=DefaultFormatter)
    parser.add_argument(
        "-f",
        "--hist-files",
        required=True,
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        action="append",
        default=[],
        help="path to input and output hist files",
    )
    args = parser.parse_args()

    for i, o in args.hist_files:
        group_hists(i, o)
