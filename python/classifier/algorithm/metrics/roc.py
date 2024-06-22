from numbers import Real
from typing import Callable, Iterable

import numpy as np
import torch
import torch.types as tt
from base_class.typetools import check_type
from torch import Tensor

from ..utils import to_arr, to_num

HistRegularAxis = tuple[int, Real, Real]


def linear_differ(pos: Tensor, neg: Tensor):
    return (pos - neg) / 2 + 0.5


class FixedThresholdROC:
    def __init__(
        self,
        thresholds: HistRegularAxis | Iterable[float],
        positive_classes: Iterable[int],
        negative_classes: Iterable[int] = None,
        score_interpretation: Callable[[Tensor, Tensor], Tensor] = None,
    ):
        self._is_regular = check_type(thresholds, HistRegularAxis)
        if self._is_regular:
            edge = thresholds
            step = (edge[2] - edge[1]) / edge[0]
            self.edge = (edge[0], step, edge[1] / step - 1)
        else:
            self.edge = np.sort(thresholds)
        self.pos = sorted(set(positive_classes))
        if negative_classes is not None:
            self.neg = sorted(set(negative_classes))
            self.map = score_interpretation
        self.reset()

    def reset(self):
        for k in ("t", "pos", "neg", "edge", "FP", "TP", "P", "N"):
            setattr(self, f"_{FixedThresholdROC.__name__}__{k}", None)

    def copy(self):
        new = super().__new__(self.__class__)
        new.__dict__ = self.__dict__.copy()
        new.reset()
        return new

    def _init(self, f: tt._dtype, i: tt._dtype, d: tt.Device):
        if self.__t is None:
            self.__t = {"dtype": f, "device": d}
            self.__pos = torch.as_tensor(self.pos, dtype=i, device=d)
            if self._has_neg:
                self.__neg = torch.as_tensor(self.neg, dtype=i, device=d)
            if not self._is_regular:
                self.__edge = torch.as_tensor(self.edge, **self.__t)
                bins = len(self.__edge) + 1
            else:
                self.__edge = torch.tensor(self.edge[1:], **self.__t)
                bins = self.edge[0] + 2
            self.__FP = torch.zeros(bins, **self.__t)
            self.__TP = torch.zeros(bins, **self.__t)
            self.__P = torch.tensor(0, **self.__t)
            self.__N = torch.tensor(0, **self.__t)

    @property
    def _has_neg(self):
        return hasattr(self, "neg")

    def _score(self, y_pred: Tensor):
        pos = y_pred[:, self.pos].sum(dim=-1)
        if (not self._has_neg) or (self.map is None):
            return pos
        else:
            neg = y_pred[:, self.neg].sum(dim=-1)
            return self.map(pos, neg)

    def _hist(self, x: Tensor, weight: Tensor, hist: Tensor):
        # -inf < b[0] < t[0] <= b[1] < t[1] <= ... < t[-1] <= b[-1] < inf
        e = self.__edge
        if self._is_regular:
            indices = torch.clip(x / e[0] - e[1], 0, self.edge[0] + 1).to(torch.int32)
        else:
            indices = torch.bucketize(x, e, right=True, out_int32=True)
        hist.index_add_(0, indices, weight)

    def _bounded(self, FPR: Tensor, TPR: Tensor):
        f, t = [FPR], [TPR]
        if FPR[0] != 1.0:
            f.insert(0, torch.tensor([1.0], **self.__t))
            t.insert(0, f[0])
        if FPR[-1] != 0.0:
            f.append(torch.tensor([0.0], **self.__t))
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
        self._init(
            weight.dtype if weight is not None else y_pred.dtype,
            y_true.dtype,
            y_pred.device,
        )
        # prepare data
        self._check_shape(y_pred=(y_pred, 2), y_true=(y_true, 1))
        if y_pred.dim() == 2:
            y_pred = self._score(y_pred)
        if weight is None:
            weight = torch.ones_like(y_true)
        else:
            self._check_shape(weight=(weight, 1), y_true=(y_true, 1))
        # update P, N, TP, FP
        p = torch.isin(y_true, self.__pos)
        n = torch.isin(y_true, self.__neg) if self._has_neg else ~p
        self.__P += weight[p].sum()
        self.__N += weight[n].sum()
        self._hist(y_pred[p], weight[p], self.__TP)
        self._hist(y_pred[n], weight[n], self.__FP)

    @torch.no_grad()
    def roc(self):
        if self.__t is None:
            return np.array([]), np.array([]), 0.0
        fpr = torch.cumsum(self.__FP, dim=0) / self.__N
        tpr = torch.cumsum(self.__TP, dim=0) / self.__P
        # deal with negative weights
        monotonic = None
        if torch.any(self.__FP < 0.0):
            monotonic = self._monotonic(fpr, monotonic)
        if torch.any(self.__TP < 0.0):
            monotonic = self._monotonic(tpr, monotonic)
        if monotonic is not None:
            fpr, tpr = fpr[monotonic], tpr[monotonic]
        # add missing (0,0) or (1,1)
        fpr, tpr = self._bounded(1 - fpr, 1 - tpr)
        # AUC
        auc = -torch.trapz(tpr, fpr)
        self.reset()
        return to_arr(fpr), to_arr(tpr), to_num(auc)
