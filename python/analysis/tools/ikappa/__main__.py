"""
Interactive plot for kappa framework analysis.

python -m analysis.tools.ikappa
"""

import json
from functools import partial
from queue import Queue
from threading import Thread
from typing import Literal

import cloudpickle
import fsspec
import yaml
from base_class.system.eos import EOS
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.models import Button, Select, Tooltip
from bokeh.models.dom import HTML
from bokeh.server.server import Server
from hist import Hist

from ._plot import Plotter, Profile
from ._sanity import sanitized
from ._utils import BokehLog, Component, PathInput, SharedDOM

_Actions = Literal["new", "attach"]


class Main(Component):
    plotter: Plotter

    _BUTTON = dict(
        button_type="success",
        sizing_mode="stretch_height",
        margin=(5, 0, 5, 5),
    )

    def __init__(self):
        super().__init__()
        # file
        self._dom_new = Button(label="New", **self._BUTTON)
        self._dom_new.on_click(partial(self._dom_load_hist, "new"))
        self._dom_attach = Button(label="Attach", **self._BUTTON)
        self._dom_attach.on_click(partial(self._dom_load_hist, "attach"))
        self._dom_hist_input = PathInput(
            title="File:",
            sizing_mode="stretch_width",
            description=Tooltip(
                content=HTML(
                    """
<b>New</b> overwrite the loaded data<br>
<b>Attach</b> extend the loaded data<br>
<b>File Extensions:</b><br>
- profiles: <code>.yml .json</code><br>
- histograms: any other files<br>
"""
                ),
                position="right",
            ),
        )
        self._dom_hist_compression = Select(
            title="Compression:",
            value="lz4",
            options=[*map(str, fsspec.compression.compr)],
        )

        # blocks
        self._file_dom = row(
            self._dom_new,
            self._dom_attach,
            self._dom_hist_input,
            self._dom_hist_compression,
            sizing_mode="stretch_width",
        )

        # data
        self._data: tuple[dict[str, Hist], set[str]] = {}, None
        self._profile: dict[str, Profile] = {}

        # async
        self._load_queue: Queue[tuple[str, _Actions]] = Queue()
        self._load_thread = Thread(target=self._load_hist, daemon=True)
        self._load_thread.start()

        self._upload_queue: Queue[tuple[str, str]] = Queue()
        self._upload_thread = Thread(target=self._upload_plot, daemon=True)
        self._upload_thread.start()

    @property
    def _hist_compression(self):
        if (comp := self._dom_hist_compression.value) == "None":
            return None
        return comp

    def _load_hist(self):
        while task := self._load_queue.get():
            path, action = task
            path = EOS(path)
            try:
                match path.extension:
                    case "yml" | "json":
                        self._ext_profile(path, action)
                    case _:
                        self._ext_data(path, action)
            except Exception as e:
                self.log.error(exec_info=e)

    def _dom_load_hist(self, action: _Actions):
        self._load_queue.put((self._dom_hist_input.value, action))

    def _upload_plot(self):
        while item := self._upload_queue.get():
            path, content = item
            try:
                with fsspec.open(path, "wt") as file:
                    self.log(f'[async] Uploading plot to "{path}"...')
                    file.write(content)
                self.log(f'[async] Uploaded to "{path}".')
            except Exception as e:
                self.log.error(exec_info=e)

    def upload(self, path: str, content: str):
        self._upload_queue.put((path, content))

    def _dom_render(self, doc: Document):
        doc.title = "i\u03BA"
        self.doc = doc
        self.log = BokehLog(doc)
        self.shared = SharedDOM(doc)
        self.plotter = Plotter(parent=self, **self.inherit_global_states)
        doc.add_root(
            column(
                self.log.dom,
                self._file_dom,
                self.plotter.dom,
                sizing_mode="stretch_both",
                margin=(0, 0, 5, 0),
            )
        )
        self.log("Ready.")

    @staticmethod
    def page(doc):
        Main()._dom_render(doc)

    def _ext_data(self, path: EOS, action: _Actions):
        if self._data[1] is None:
            action = "new"
        with fsspec.open(path, mode="rb", compression=self._hist_compression) as file:
            self.log(f'[async] Loading data from "{path}"...')
            data = cloudpickle.load(file)
            self.log(f'[async] Loaded from "{path}".')
            hists, categories = data["hists"], data["categories"]
            match action:
                case "new":
                    self.log("Creating new workspace.")
                    self._data = hists, categories
                case "attach":
                    self.log("Attaching to existing workspace.")
                    if self._data[1] != categories:
                        raise ValueError("Category mismatch.")
                    self._data[0].update(hists)
            self.log("Sanitizing data...")
            groups = sanitized(*self._data)
            skipped = sum(groups[1:], [])
            if skipped:
                self.log.error(
                    "The following histograms are skipped because of category mismatch",
                    *skipped,
                )
            self._data = {k: self._data[0][k] for k in groups[0]}, self._data[1]
            self.log("Loading plotter...")
            self.plotter.update_data(*self._data)
        self.log("Done.")

    def _ext_profile(self, path: EOS, action: _Actions):
        with fsspec.open(path, mode="rt") as file:
            self.log(f'[async] Loading profile from "{path}"...')
            data: dict[str, Profile]
            match ext := path.extension:
                case "yml":
                    data = yaml.safe_load(file)
                case "json":
                    data = json.load(file)
                case _:
                    raise ValueError(f'Unsupported file format "{ext}".')
            match action:
                case "new":
                    self.log("Applying new profile.")
                    self._profile = data
                case "attach":
                    self.log("Attaching to existing profile.")
                    self._profile.update(data)
            self.log("Updating plotter...")
            self.plotter.update_profile(self._profile)
        self.log("Done.")


if __name__ == "__main__":
    server = Server(
        {"/": Main.page},
        num_procs=1,
    )
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
