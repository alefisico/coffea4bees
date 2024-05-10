from typing import Iterable

import torch
import torch.types as tt

from ..utils import to_arr, to_num


class ROCFixedThreshold:
    _value_type = torch.float32
    _index_type = torch.uint8

    def __init__(
        self,
        pos: Iterable[int],
        neg: Iterable[int],
        threshold: Iterable[float],
        device: tt.Device = None,
    ):
        self.__f = {"dtype": self._value_type, "device": device}
        self.__i = {"dtype": self._index_type, "device": device}
        self._pos = torch.as_tensor(pos, **self.__i)
        self._neg = torch.as_tensor(neg, **self.__i)
        self._threshold, _ = torch.sort(torch.as_tensor(threshold, **self.__f))
        self.reset()

    def reset(self):
        self._FP = torch.zeros(len(self._threshold) + 1, **self.__f)
        self._TP = torch.zeros(len(self._threshold) + 1, **self.__f)
        self._P = torch.tensor(0, **self.__f)
        self._N = torch.tensor(0, **self.__f)

    def _hist(self, x: torch.Tensor, weight: torch.Tensor, hist: torch.Tensor):
        # -inf < b[0] < t[0] <= b[1] < t[1] <= ... < t[-1] <= b[-1] < inf
        hist.index_add_(0, torch.searchsorted(self._threshold, x, right=True), weight)

    @staticmethod
    def _bounded(FPR: torch.Tensor, TPR: torch.Tensor):
        f, t = [FPR], [TPR]
        if FPR[0] != 1.0:
            f.insert(0, torch.tensor([1.0], dtype=FPR.dtype))
            t.insert(0, f[0])
        if FPR[-1] != 0.0:
            f.append(torch.tensor([0.0], dtype=FPR.dtype))
            t.append(f[-1])
        if len(f) > 1:
            return torch.cat(f), torch.cat(t)
        return FPR, TPR

    @staticmethod
    def _check_shape(**tensors):
        sizes = {k: len(v) for k, v in tensors.items()}
        if not len(set(sizes.values())) == 1:
            msg = ", ".join(f"{k}({v})" for k, v in sizes.items())
            raise ValueError(f"{msg} must have the same length")

    @torch.no_grad()
    def update(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None
    ):
        # prepare data
        y_pred = torch.as_tensor(y_pred, **self.__f)
        y_true = torch.as_tensor(y_true, **self.__i)
        self._check_shape(y_pred=y_pred, y_true=y_true)
        if weight is None:
            weight = torch.ones_like(y_true, **self.__f)
        else:
            weight = torch.as_tensor(weight, **self.__f)
            self._check_shape(weight=weight, y_pred=y_pred, y_true=y_true)
        # update P, N, TP, FP
        p = torch.isin(y_true, self._pos)
        n = torch.isin(y_true, self._neg)
        self._P += weight[p].sum()
        self._N += weight[n].sum()
        self._hist(y_pred[p], weight[p], self._TP)
        self._hist(y_pred[n], weight[n], self._FP)

    @torch.no_grad()
    def roc(self):
        FPR = torch.cumsum(self._FP, dim=0) / self._N
        TPR = torch.cumsum(self._TP, dim=0) / self._P
        # deal with negative weights
        if torch.any(self._FP < 0.0):
            monotonic = FPR >= torch.cummax(FPR, dim=0)[0]
            FPR, TPR = FPR[monotonic], TPR[monotonic]
        # add missing (0,0) or (1,1)
        FPR, TPR = self._bounded(1 - FPR, 1 - TPR)
        # AUC
        AUC = -torch.trapz(TPR, FPR)
        self.reset()
        return to_arr(FPR), to_arr(TPR), to_num(AUC)
