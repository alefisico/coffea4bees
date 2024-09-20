import logging
from argparse import ArgumentParser

import cloudpickle
import fsspec
import pandas as pd
from base_class.system.eos import EOS
from rich.logging import RichHandler

from ._sanity import group_by_categories, group_to_str

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=False, show_path=False)],
    )
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="input file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="output file",
    )
    args = parser.parse_args()

    with fsspec.open(args.input, "rb", compression="lz4") as f:
        data = cloudpickle.load(f)

    categories, groups = group_by_categories(data["hists"], data["categories"])
    if failed := groups.pop(None, None):
        logging.warning(f"The following histograms have missing categories: {failed}")
    axes = pd.DataFrame(groups.keys(), columns=categories)
    axes = axes.loc[:, axes.nunique(axis=0) > 1]
    output = EOS(args.output)
    for i, group in enumerate(groups.values()):
        g = i + 1
        file = output.parent / f"{output.stem}.{g}.coffea"
        axis = axes.iloc[i]
        logging.info(f"Group {g}: {file}")
        for k, v in axis.items():
            logging.info(f"{k}: {group_to_str(v)}")
        logging.info("histograms: " + ", ".join(group))
        with fsspec.open(file, "wb", compression="lz4") as f:
            cloudpickle.dump(
                {
                    "hists": {k: data["hists"][k] for k in group},
                    "categories": data["categories"],
                },
                f,
            )
