from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.setting.dataframe import Columns
from ..config.state.label import MultiClass

if TYPE_CHECKING:
    import pandas as pd


class add_label_index:
    def __init__(self, label: str):
        self.label = label

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        df[Columns.label_index] = np.dtype(
            Columns.index_dtype).type(MultiClass.index(self.label))
        return df


class add_label_flag:
    def __init__(self, *labels: str):
        self.labels = {*labels}

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for label in MultiClass.labels:
            df[label] = label in self.labels
        return df


class add_event_offset:
    """
    a workaround for no ``uint64`` in torch.Tensor
    """

    def __init__(self, kfold: int):
        self.kfold = kfold

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df[Columns.event_offset] = (
            df[Columns.event] % self.kfold).astype(Columns.index_dtype)
        return df


def normalize_weight(df: pd.DataFrame) -> pd.DataFrame:
    df[Columns.weight_normalized] = df[Columns.weight] / df[Columns.weight].sum()
    return df
