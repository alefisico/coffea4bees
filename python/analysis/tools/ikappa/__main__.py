"""
Interactive plot for kappa framework analysis.

python -m analysis.tools.ikappa
"""

import json
from argparse import ArgumentParser
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
from ._utils import BokehLog, Component, ExternalLink, PathInput, SharedDOM

_Actions = Literal["new", "add"]
_INDENT = "  "


class _ProfileStyle(yaml.SafeDumper):
    @classmethod
    def setup(cls):
        cls.add_representer(dict, cls.represent_dict)
        cls.add_representer(list, cls.represent_list)

    def represent_dict(self, data):
        return self.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)

    def represent_list(self, data):
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


_ProfileStyle.setup()


class Main(Component):
    plotter: Plotter

    _BUTTON = dict(
        sizing_mode="stretch_height",
        margin=(5, 0, 5, 5),
    )

    def __init__(self):
        super().__init__()
        # data
        self._hists: tuple[dict[str, Hist], set[str]] = {}, None
        self._profiles: dict[str, Profile] = {}
        self._status = {"Histograms": [], "Profiles": []}

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

    def _dom_show_status(self):
        content = []
        content.append("<b>Files</b>:<br>")
        for category, paths in self._status.items():
            if not paths:
                continue
            content.append(f"{_INDENT}{category}:<br>")
            content.extend(f"{_INDENT*2}- {path}<br>" for path in paths)
        if self._hists[0] and self._hists[1] is not None:
            content.append("<b>Categories</b>:<br>")
            content.append("<div class='itemize box'>")
            for k, v in self.plotter.categories.items():
                content.append(f"{_INDENT}{k}")
                content.append("<div class='code'>")
                content.append("<br>".join(f"{_INDENT*2}{bin}" for bin in v._choices))
                content.append("</div>")
            content.append("</div>")
            content.append("<b>Histograms</b>:<br>")
            content.append("<div class='itemize box'>")
            for name, hist in self._hists[0].items():
                content.append(f"{_INDENT}{name}")
                content.append("<div class='code'>")
                content.append(
                    "<br>".join(
                        f"{_INDENT*2}{axis}"
                        for axis in hist.axes
                        if axis.name not in self._hists[1]
                    )
                )
                content.append("</div>")
            content.append("</div>")
        if self._profiles:
            content.append("<b>Profiles</b>:<br>")
            content.append("<div class='code box'>")
            content.append(
                yaml.dump(self._profiles, Dumper=_ProfileStyle).replace("\n", "<br>")
            )
            content.append("</div>")

        return "".join(content)

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
        # file
        self._dom_new = Button(label="New", button_type="warning", **self._BUTTON)
        self._dom_new.on_click(partial(self._dom_load_hist, "new"))
        self._dom_add = Button(label="Add", button_type="success", **self._BUTTON)
        self._dom_add.on_click(partial(self._dom_load_hist, "add"))
        self._dom_status = ExternalLink(
            shared=self.shared, label="Status", button_type="primary", **self._BUTTON
        )
        self._dom_status.add_page(
            self._dom_show_status,
            """         
text = `<!DOCTYPE html><html><head><title>Status</title><style>
div.kappa-framework {white-space: pre;}
div.box {background-color: #f0f0f0; border: 1px solid #d0d0d0; width: fit-content; min-width: calc(100% - 20px); padding: 10px;}
div.code {font-family: monospace; white-space: pre; display: block;}
div.itemize {white-space: break-spaces;}
</style></head><body><div class="kappa-framework" style="width: auto;">`+ text + "</div></body></html>"
""",
        )
        self._dom_hist_input = PathInput(
            title="File:",
            sizing_mode="stretch_width",
            description=Tooltip(
                content=HTML(
                    """
<b>New</b> overwrite the loaded data<br>
<b>Add</b> extend the loaded data<br>
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
            self._dom_add,
            self._dom_hist_input,
            self._dom_hist_compression,
            *self._dom_status,
            sizing_mode="stretch_width",
        )
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

    def _update_status(self, *status: str, category: str, action: _Actions):
        match action:
            case "new":
                self._status[category] = [*status]
            case "add":
                self._status[category].extend(status)

    def _ext_data(self, path: EOS, action: _Actions):
        if self._hists[1] is None:
            action = "new"
        with fsspec.open(path, mode="rb", compression=self._hist_compression) as file:
            self.log(f'[async] Loading data from "{path}"...')
            data = cloudpickle.load(file)
            self.log(f'[async] Loaded from "{path}".')
            hists, categories = data["hists"], data["categories"]
            match action:
                case "new":
                    self.log("Creating new workspace.")
                    self._hists = hists, categories
                case "add":
                    self.log("Adding to existing workspace.")
                    if self._hists[1] != categories:
                        raise ValueError("Category mismatch.")
                    self._hists[0].update(hists)
            self.log("Sanitizing data...")
            groups = sanitized(*self._hists)
            skipped = sum(groups[1:], [])
            if skipped:
                self.log.error(
                    "The following histograms are skipped because of category mismatch",
                    *skipped,
                )
            self._hists = {k: self._hists[0][k] for k in groups[0]}, self._hists[1]
            self.log("Loading plotter...")
            self.plotter.update_data(*self._hists)
        self._update_status(str(path), category="Histograms", action=action)
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
                    self._profiles = data
                case "add":
                    self.log("Adding to existing profile.")
                    self._profiles.update(data)
            self.log("Updating plotter...")
            self.plotter.update_profile(self._profiles)
        self._update_status(str(path), category="Profiles", action=action)
        self.log("Done.")


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-p", "--port", type=int, default=10200)
    args = argparser.parse_args()

    server = Server(
        {"/": Main.page},
        num_procs=1,
        port=args.port,
    )
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
