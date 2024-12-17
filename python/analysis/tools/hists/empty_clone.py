from argparse import ArgumentParser

import cloudpickle
import fsspec
from base_class.utils.argparser import DefaultFormatter
from hist import Hist


def clone_empty(in_path: str, out_path: str):
    with fsspec.open(in_path, "rb", compression="lz4") as f:
        data: dict[str, dict[str, Hist]] = cloudpickle.load(f)
    data["hists"] = {
        k: Hist(*v.axes, storage=v.storage_type()) for k, v in data["hists"].items()
    }
    with fsspec.open(out_path, "wb", compression="lz4") as f:
        cloudpickle.dump(data, f)


if __name__ == "__main__":
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
        clone_empty(i, o)
