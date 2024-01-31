import numpy as np
import hist
from hist.axis import Regular
from sklearn.metrics import roc_curve, auc
from ..dataset import Label, LabelCollection
import pandas as pd


def roc_auc_with_negative_weights(
        classes: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray = None) -> float:
    """
    Calculating ROC AUC score as the probability of correct ordering
    Credit: https://github.com/SiLiKhon/my_roc_auc/blob/master/my_roc_auc.py
    """

    if weights is None:
        weights = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(weights)
    assert classes.ndim == predictions.ndim == weights.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
        shape=len(classes),
        dtype=[('c', classes.dtype),
               ('p', predictions.dtype),
               ('w', weights.dtype)]
    )
    data['c'], data['p'], data['w'] = classes, predictions, weights

    data = data[np.argsort(data['c'])]
    # here we're relying on stability as we need class orders preserved
    data = data[np.argsort(data['p'], kind='mergesort')]

    correction = 0.
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data['p'][1:] == data['p'][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        ids, = mask2.nonzero()
        correction = sum([((dsplit['c'] == class0) * dsplit['w'] * msplit).sum() *
                          ((dsplit['c'] == class1) *
                           dsplit['w'] * msplit).sum()
                          for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))]) * 0.5

    weights_0 = data['w'] * (data['c'] == class0)
    weights_1 = data['w'] * (data['c'] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (weights_1.sum() * cumsum_0[-1])


class ROC:  # TODO rewrite
    steps = Regular(20, 0, 1, 'y')

    def __init__(
            self,
            true: Label,
            false: Label,
            title: str):
        self.bins = np.arange(0, 1.05, 0.05)
        self.S, _ = pltHelper.binData(
            y_pred[y_true == 1], self.bins, weights=weights[y_true == 1])
        self.B, _ = pltHelper.binData(
            y_pred[y_true == 0], self.bins, weights=weights[y_true == 0])
        self.fpr, self.tpr, self.thr = roc_curve(
            y_true, y_pred, sample_weight=weight_train)
        self.auc = roc_auc_with_negative_weights(
            y_true, y_pred, weights=weight_train)
        self.sigma = None
        if "Background" in self.falseName:
            wS = None
            wB = None
            lumiRatio = 1  # 140/59.6
            if 'Signal Region' in title:
                wB = sum_wB_SR
            else:
                wB = sum_wB

            if "ZZ" in self.trueName:
                wS = wzz_SR
                self.pName = "P(Signal)"
            if "ZH" in self.trueName:
                wS = wzh_SR
                self.pName = "P(Signal)"
            if "HH" in self.trueName:
                wS = whh_SR
                self.pName = "P(Signal)"
            if "Signal" in self.trueName:
                wS = sum_wS_SR
                self.pName = "P(Signal)"

            self.S = wS*lumiRatio * self.S/self.S.sum()
            self.B = wB*lumiRatio * self.B/self.B.sum()
            # include 3% background systematic and 10% signal systematic and \sqrt{5} event fixed uncertainty
            sigma = self.S / np.sqrt(self.S+self.B +
                                     (0.03*self.B)**2 + (0.1*self.S)**2 + 5)
            self.sigma = np.sqrt((sigma**2).sum())
            self.tprSigma = self.S[-1:].sum() / self.S.sum()
            self.fprSigma = self.B[-1:].sum() / self.B.sum()
            self.thrSigma = self.bins[-2]

            self.S = self.S[-1]
            self.B = self.B[-1]
