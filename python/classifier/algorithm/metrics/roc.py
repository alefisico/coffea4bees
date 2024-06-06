from typing import Callable, Iterable

import torch
import torch.types as tt
from torch import Tensor

from ..utils import to_arr, to_num


def linear_differ(pos: Tensor, neg: Tensor):
    return (pos - neg) / 2 + 0.5


class FixedThresholdROC:
    _value_type = torch.float32
    _index_type = torch.uint8

    def __init__(
        self,
        thresholds: Iterable[float],
        positive_classes: Iterable[int],
        negative_classes: Iterable[int] = None,
        score_interpretation: Callable[[Tensor, Tensor], Tensor] = None,
        device: tt.Device = None,
    ):
        self.__f = {"dtype": self._value_type, "device": device}
        self.__i = {"dtype": self._index_type, "device": device}
        # classes
        self._pos = self._classes(positive_classes)
        if negative_classes is not None:
            self._neg = self._classes(negative_classes)
            self._map = score_interpretation
        # count
        self._edge, _ = torch.sort(torch.as_tensor(thresholds, **self.__f))
        self.reset()

    def reset(self):
        self._FP = torch.zeros(len(self._edge) + 1, **self.__f)
        self._TP = torch.zeros(len(self._edge) + 1, **self.__f)
        self._P = torch.tensor(0, **self.__f)
        self._N = torch.tensor(0, **self.__f)

    @property
    def __neg(self):
        return hasattr(self, "_neg")

    def _classes(self, classes: Iterable[int]):
        classes = sorted(set(classes))
        return torch.as_tensor(classes, **self.__i), classes

    def _score(self, y_pred: Tensor):
        pos = y_pred[:, self._pos[1]].sum(dim=-1)
        if (not self.__neg) or (self._map is None):
            return pos
        else:
            neg = y_pred[:, self._neg[1]].sum(dim=-1)
            return self._map(pos, neg)

    def _hist(self, x: Tensor, weight: Tensor, hist: Tensor):
        # -inf < b[0] < t[0] <= b[1] < t[1] <= ... < t[-1] <= b[-1] < inf
        hist.index_add_(0, torch.searchsorted(self._edge, x, right=True), weight)

    def _bounded(self, FPR: Tensor, TPR: Tensor):
        f, t = [FPR], [TPR]
        if FPR[0] != 1.0:
            f.insert(0, torch.tensor([1.0], **self.__f))
            t.insert(0, f[0])
        if FPR[-1] != 0.0:
            f.append(torch.tensor([0.0], **self.__f))
            t.append(f[-1])
        if len(f) > 1:
            return torch.cat(f), torch.cat(t)
        return FPR, TPR

    @staticmethod
    def _monotonic(new: Tensor, old: Tensor):
        new = new >= torch.cummax(new, dim=0)[0]
        if old is not None:
            new &= old
        return new

    @staticmethod
    def _check_shape(**tensors: tuple[Tensor, int]):
        sizes = {}
        for k, (v, dim) in tensors.items():
            if v.dim() > dim:
                raise ValueError(f"{k} must have at most {dim} dimensions")
            sizes[k] = len(v)
        if not len(set(sizes.values())) == 1:
            msg = ", ".join(f"{k}({v})" for k, v in sizes.items())
            raise ValueError(f"{msg} must have the same length")

    @torch.no_grad()
    def update(self, y_pred: Tensor, y_true: Tensor, weight: Tensor = None):
        # prepare data
        y_pred = torch.as_tensor(y_pred, **self.__f)
        y_true = torch.as_tensor(y_true, **self.__i)
        self._check_shape(y_pred=(y_pred, 2), y_true=(y_true, 1))
        if y_pred.dim() == 2:
            y_pred = self._score(y_pred)
        if weight is None:
            weight = torch.ones_like(y_true, **self.__f)
        else:
            weight = torch.as_tensor(weight, **self.__f)
            self._check_shape(weight=(weight, 1), y_true=(y_true, 1))
        # update P, N, TP, FP
        p = torch.isin(y_true, self._pos[0])
        n = torch.isin(y_true, self._neg[0]) if self.__neg else ~p
        self._P += weight[p].sum()
        self._N += weight[n].sum()
        self._hist(y_pred[p], weight[p], self._TP)
        self._hist(y_pred[n], weight[n], self._FP)

    @torch.no_grad()
    def roc(self):
        fpr = torch.cumsum(self._FP, dim=0) / self._N
        tpr = torch.cumsum(self._TP, dim=0) / self._P
        # deal with negative weights
        monotonic = None
        if torch.any(self._FP < 0.0):
            monotonic = self._monotonic(fpr, monotonic)
        if torch.any(self._TP < 0.0):
            monotonic = self._monotonic(tpr, monotonic)
        if monotonic is not None:
            fpr, tpr = fpr[monotonic], tpr[monotonic]
        # add missing (0,0) or (1,1)
        fpr, tpr = self._bounded(1 - fpr, 1 - tpr)
        # AUC
        auc = -torch.trapz(tpr, fpr)
        self.reset()
        return to_arr(fpr), to_arr(tpr), to_num(auc)
