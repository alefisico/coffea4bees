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
        with fsspec.open(model, "rb") as f:
            states = torch.load(f, map_location=torch.device("cpu"))
        self._classes: list[str] = states["label"]
        self._reindex: list[int] = None
        self._ancillary = states["arch"]["ancillary"]
        self._model = HCR(
            dijetFeatures=states["arch"]["n_features"],
            quadjetFeatures=states["arch"]["n_features"],
            ancillaryFeatures=self._ancillary,
            useOthJets=("attention" if states["arch"]["attention"] else ""),
            nClasses=len(self._classes),
        )
        self._model.load_state_dict(states["model"], map_location=torch.device("cpu"))
        self.splitter = splitter

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

    def __call__(self, event: ak.Array): ...  # TODO


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
        self.ancillary = self.models[0]._ancillary
        for model in self.models:
            if model._ancillary != self.ancillary:
                raise ValueError(
                    f"HCR evaluation: ancillary features mismatch, expected {self.ancillary} got {model._ancillary}"
                )
            model.classes = self.classes

    def __call__(self, event: ak.Array) -> tuple[npt.NDArray, npt.NDArray]: ...  # TODO
