"""
Interactive plot for kappa framework analysis.

python -m analysis.tools.ikappa
"""

from queue import Queue
from threading import Thread

import cloudpickle
import fsspec
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.models import Button, Select
from bokeh.server.server import Server

from ._plot import Plotter
from ._utils import BokehLog, PathInput


class Main:
    log: BokehLog
    plotter: Plotter

    def __init__(self):
        # file
        self._dom_hist_submit = Button(
            label="Load", button_type="success", sizing_mode="stretch_height"
        )
        self._dom_hist_input = PathInput(title="File:", sizing_mode="stretch_width")
        self._dom_hist_compression = Select(
            title="Compression:",
            value="lz4",
            options=[*map(str, fsspec.compression.compr)],
        )
        self._dom_hist_submit.on_click(self._dom_load_hist)

        # blocks
        self._file_dom = row(
            self._dom_hist_submit,
            self._dom_hist_input,
            self._dom_hist_compression,
            sizing_mode="stretch_width",
        )
        self.dom = column(sizing_mode="stretch_both", margin=(0, 0, 5, 0))

        self._load_queue = Queue()
        self._load_thread = Thread(target=self._load_hist, daemon=True)
        self._load_thread.start()

    @property
    def full(self):
        return self._full

    @full.setter
    def full(self, value):
        self._full = value
        if value:
            self.dom.children = [self.plotter.dom]
        else:
            self.dom.children = [self.log.dom, self._file_dom, self.plotter.dom]

    @property
    def _hist_compression(self):
        if (comp := self._dom_hist_compression.value) == "None":
            return None
        return comp

    def _load_hist(self):
        while path := self._load_queue.get():
            try:
                with fsspec.open(path, compression=self._hist_compression) as file:
                    self.log(f'Loading data from "{path}"...')
                    data = cloudpickle.load(file)
                    self.log(f'Data loaded from "{path}"')
                    self.plotter.update(data["hists"], data["categories"])
            except Exception as e:
                self.log.error(exec_info=e)

    def _dom_load_hist(self):
        self._load_queue.put(self._dom_hist_input.value)

    def _dom_render(self, doc: Document):
        doc.title = "i\u03BA"
        self.log = BokehLog(doc)
        self.plotter = Plotter(doc, self.log, self)
        self.full = False
        doc.add_root(self.dom)
        self.log("Ready.")

    @staticmethod
    def page(doc):
        Main()._dom_render(doc)


if __name__ == "__main__":
    server = Server(
        {"/": Main.page},
        num_procs=1,
    )
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
