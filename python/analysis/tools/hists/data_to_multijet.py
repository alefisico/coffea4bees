import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import repeat

import cloudpickle
import fsspec
from base_class.system.eos import EOS, PathLike
from base_class.utils.argparser import DefaultFormatter
from hist import Hist
from rich.logging import RichHandler

from ..ikappa._bh import BHAxis, HistAxis


def path_from_pattern(pattern: str, path: PathLike):
    """
    Notes
    -----
    The following keys are available in the pattern for the example path:

    `"root://host.url//path/to/file.root"`

    - `{host}`: `"root://host.url/"`
    - `{name}`: `"file"`
    - `{ext}`: `"root"`
    - `{part1}`: `"to"`
    - `{part2}`: `"path"`
    - ...
    - `{parent1}`: `"/path/to"`
    - `{parent2}`: `"/path"`
    - ...
    """
    path = EOS(path)
    parent = path.path.parent
    parents = {}
    parts = {}
    while parent.parent != parent:
        parents[f"parent{len(parents) + 1}"] = str(parent)
        parts[f"part{len(parts) + 1}"] = parent.name
        parent = parent.parent

    return EOS(
        pattern.format(
            name=path.stem,
            ext=path.extension,
            host=path.host,
            **parts,
            **parents,
        )
    )


@dataclass(kw_only=True)
class Data3bToMultijet4b:
    output: str
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
            logging.warning(f"Missing axis in hist {name}, skipping")
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

    def __call__(self, input: str):
        output = path_from_pattern(self.output, input)
        with fsspec.open(input, mode="rb", compression="lz4") as f:
            data: dict[str, dict[str, Hist]] = cloudpickle.load(f)
        data = {
            "hists": {
                k: h
                for k, v in data["hists"].items()
                if (h := self.extend(v, k)) is not None
            },
            "categories": data["categories"],
        }
        with fsspec.open(output, mode="wb", compression="lz4") as f:
            cloudpickle.dump(data, f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_time=False, show_path=False, markup=True)],
    )
    parser = ArgumentParser(formatter_class=DefaultFormatter)
    parser.add_argument(
        "-i",
        "--input-files",
        required=True,
        help="path to input hist files",
        nargs="+",
        action="extend",
        default=[],
    )
    parser.add_argument(
        "-o",
        "--output-pattern",
        help="output path pattern (see documentation for path_from_pattern)",
        default="{host}{parent1}/{name}_mj.{ext}",
    )
    parser.add_argument(
        "-p",
        "--processes",
        nargs=2,
        metavar=("data", "multijet"),
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
        output=args.output_pattern,
        bins=args.processes,
        ax_process=args.process_axis,
        ax_tag=args.tag_axis,
    )

    for input in args.input_files:
        d3tomj4(input)
