import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import repeat

import cloudpickle
import fsspec
from base_class.utils.argparser import DefaultFormatter
from hist import Hist
from rich.logging import RichHandler

from ..ikappa._bh import BHAxis, HistAxis


@dataclass(kw_only=True)
class Data3bToMultijet4b:
    bins: list[tuple[str, str]]
    ax_process: str
    ax_tag: str

    def __post_init__(self):
        self.bh = BHAxis(flow=True)

    def _extend(self, hist: Hist, name: str, data: str, multijet: str) -> Hist:
        axes: list[HistAxis] = [*hist.axes]
        # find axes
        i_process = None
        i_tag = None
        for i, ax in enumerate(axes):
            if ax.name == self.ax_process:
                i_process = i
            elif ax.name == self.ax_tag:
                i_tag = i
        if i_process is None or i_tag is None:
            logging.warning(f"Missing axis in hist {name}, skipped")
            return None
        # copy hist
        ax_process = axes[i_process]
        n_process = len(ax_process)
        axes[i_process] = self.bh.extend(ax_process, multijet)
        slicers = [*repeat(slice(None), len(axes))]
        slicers[i_process] = slice(0, n_process)
        new = Hist(*axes, storage=hist.storage_type())
        new_view = new.view(flow=True)
        old_view = hist.view(flow=True)
        new_view[(*slicers,)] = old_view[(*slicers,)]
        if not ax_process.traits.growth:
            new_slicers = slicers.copy()
            new_slicers[i_process] = slice(n_process + 1, n_process + 2)
            old_slicers = slicers.copy()
            old_slicers[i_process] = slice(n_process, n_process + 1)
            new_view[(*new_slicers,)] = old_view[(*old_slicers,)]
        # copy 3b data to 4b multijet
        new_slicers = slicers.copy()
        old_slicers = slicers.copy()
        for i, process in enumerate(axes[i_process]):
            if process == multijet:
                new_slicers[i_process] = i
            elif process == data:
                old_slicers[i_process] = i
        for i, tag in enumerate(axes[i_tag]):
            if tag == 4:
                new_slicers[i_tag] = i
            elif tag == 3:
                old_slicers[i_tag] = i
        new_view[(*new_slicers,)] = old_view[(*old_slicers,)]
        return new

    def extend(self, hist: Hist, name: str):
        for data, multijet in self.bins:
            hist = self._extend(hist, name, data, multijet)
        return hist

    def __call__(self, in_path: str, out_path: str):
        with fsspec.open(in_path, mode="rb", compression="lz4") as f:
            data: dict[str, dict[str, Hist]] = cloudpickle.load(f)
        data["hists"] = {
            k: h
            for k, v in data["hists"].items()
            if (h := self.extend(v, k)) is not None
        }
        with fsspec.open(out_path, mode="wb", compression="lz4") as f:
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
    parser.add_argument(
        "-p",
        "--processes",
        required=True,
        nargs=2,
        metavar=("DATA", "MULTIJET"),
        action="append",
        default=[],
        help="name of data and multijet processes",
    )
    parser.add_argument(
        "--process-axis",
        default="process",
        help="name of process axis",
    )
    parser.add_argument(
        "--tag-axis",
        default="tag",
        help="name of tag axis",
    )

    args = parser.parse_args()

    d3tomj4 = Data3bToMultijet4b(
        bins=args.processes,
        ax_process=args.process_axis,
        ax_tag=args.tag_axis,
    )

    for i, o in args.hist_files:
        d3tomj4(i, o)
