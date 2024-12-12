from argparse import Namespace

import awkward as ak
import fsspec
import numpy.typing as npt
import torch
import torch.nn.functional as F
from base_class.system.eos import EOS, PathLike
from classifier.config.model._kfold import KFoldEval
from classifier.ml.skimmer import Splitter
from classifier.nn.blocks.HCR import HCR
from classifier.ml import BatchType
from classifier.config.setting.HCR import Input
from classifier.config.setting.ml import SplitterKeys, KFold

from .. import networks


class _Legacy_HCREnsemble(networks.HCREnsemble):
    classes = ["multijet", "ttbar", "ZZ", "ZH", "ggF"]

    def __call__(self, event: ak.Array):
        n = len(event)
        # candidate jet features
        j = torch.zeros(n, 4, 4)
        for i, k in enumerate(("pt", "eta", "phi", "mass")):
            j[:, i, :] = torch.tensor(event.canJet[k])
        # other jet features
        o = torch.zeros(n, 5, 8)
        for i, k in enumerate(("pt", "eta", "phi", "mass", "isSelJet")):
            o[:, i, :] = torch.tensor(
                ak.fill_none(
                    ak.to_regular(
                        ak.pad_none(event.notCanJet_coffea[k], target=8, clip=True)
                    ),
                    -1,
                )
            )
        # ancillary features
        a = torch.zeros(n, 4)
        a[:, 0] = float(event.metadata["year"][3])
        a[:, 1] = torch.tensor(event.nJet_selected)
        a[:, 2] = torch.tensor(event.xW)
        a[:, 3] = torch.tensor(event.xbW)
        # event offset
        e = torch.tensor(event.event) % 3

        c_logits, q_logits = self.forward(j, o, a, e)
        return F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy()


class _HCRKFoldModel:
    def __init__(self, model: str, splitter: Splitter):
        self.splitter = splitter
        with fsspec.open(model, "rb") as f:
            states = torch.load(f, map_location=torch.device("cpu"))
        self.ancillary = states["input"]["feature_ancillary"]
        self.n_othjets = states["input"]["n_NotCanJet"]

        self._classes: list[str] = states["label"]
        self._reindex: list[int] = None
        self._model = HCR(
            dijetFeatures=states["arch"]["n_features"],
            quadjetFeatures=states["arch"]["n_features"],
            ancillaryFeatures=self.ancillary,
            useOthJets=("attention" if states["arch"]["attention"] else ""),
            nClasses=len(self._classes),
        )
        self._model.load_state_dict(states["model"], map_location=torch.device("cpu"))

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        if set(value) <= set(self._classes):
            if value != self._classes:
                self._reindex = [self._classes.index(c) for c in value]
        else:
            raise ValueError(
                f"HCR evaluation: classes mismatch, unknown classes: {set(value) - set(self._classes)}"
            )

    @property
    def eval(self):
        return self

    def __call__(self, j, o, a):
        c_logits, q_logits = self._model(j, o, a)
        if self._reindex is not None:
            c_logits = c_logits[:, self._reindex]
        return c_logits, q_logits


class _HCRKFoldEval(KFoldEval):
    def __init__(self, path: PathLike, name: str):
        self.opts = Namespace(model=[(name, path)])

    def initializer(self, model: str, splitter: Splitter, **_):
        return _HCRKFoldModel(model, splitter)


class HCREnsemble:
    def __new__(cls, path: PathLike, name: str = None):
        match EOS(path).extension:
            case "pkl":
                return _Legacy_HCREnsemble(path)
            case "json":
                return cls(path, name)

    def __init__(self, path: PathLike, name: str = None):
        self.models: list[_HCRKFoldModel] = _HCRKFoldEval(path, name).evaluate()
        self.classes = self.models[0].classes
        self.ancillary = self.models[0].ancillary
        self.n_othjets = self.models[0].n_othjets
        for model in self.models:
            for k in ("ancillary", "n_othjets"):
                if getattr(self, k) != getattr(model, k):
                    raise ValueError(
                        f"HCR evaluation: {k} mismatch, expected {getattr(self, k)} got {getattr(model, k)}"
                    )
            model.classes = self.classes

    def __call__(self, event: ak.Array) -> tuple[npt.NDArray, npt.NDArray]:
        n = len(event)
        batch: BatchType = {
            Input.CanJet: torch.zeros(n, 4, 4, dtype=torch.float32),
            Input.NotCanJet: torch.zeros(n, 5, self.n_othjets, dtype=torch.float32),
            Input.ancillary: torch.zeros(n, len(self.ancillary), dtype=torch.float32),
        }
        # candidate jet features
        j = batch[Input.CanJet]
        for i, k in enumerate(("pt", "eta", "phi", "mass")):
            j[:, i, :] = torch.tensor(event.canJet[k])
        # other jet features
        o = batch[Input.NotCanJet]
        for i, k in enumerate(("pt", "eta", "phi", "mass", "isSelJet")):
            o[:, i, :] = torch.tensor(
                ak.fill_none(
                    ak.to_regular(
                        ak.pad_none(
                            event.notCanJet_coffea[k], target=self.n_othjets, clip=True
                        )
                    ),
                    -1,
                )
            )
        # ancillary features
        a = batch[Input.ancillary]
        for i, k in enumerate(self.ancillary):
            match k:
                case "year":
                    a[:, i] = float(event.metadata["year"][-2:])
                case "nSelJets":
                    a[:, i] = torch.tensor(event.nJet_selected)
                case "xW":
                    a[:, i] = torch.tensor(event.xW)
                case "xbW":
                    a[:, i] = torch.tensor(event.xbW)
        # event offset
        batch[KFold.offset] = torch.tensor(event.event, dtype=KFold.offset_dtype)

        c_logits = torch.zeros(n, len(self.classes), dtype=torch.float32)
        q_logits = torch.zeros(n, 3, dtype=torch.float32)
        for model in self.models:
            mask = model.splitter.split(batch)[SplitterKeys.validation]
            c_logits[mask], q_logits[mask] = model(j[mask], o[mask], a[mask])

        return F.softmax(c_logits, dim=-1).numpy(), F.softmax(q_logits, dim=-1).numpy()
