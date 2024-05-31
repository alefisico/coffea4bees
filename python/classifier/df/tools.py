from __future__ import annotations

from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Iterable, Literal

from base_class.math.random import SeedLike, Squares

from ..config.setting.df import Columns
from ..config.state.label import MultiClass
from ..utils import keep_fraction

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd


class add_label_index:
    def __init__(self, label: str):
        MultiClass.add(label)
        self._label = label

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        df.loc[:, (Columns.label_index,)] = np.dtype(Columns.index_dtype).type(
            MultiClass.index(self._label)
        )
        return df


class add_label_index_from_column:
    def __init__(self, **labels: str):
        MultiClass.add(*labels.values())
        self._labels = labels

    @cached_property
    def _calc(self):
        return map_selection_to_index(
            **{k: MultiClass.index(v) for k, v in self._labels.items()}
        ).set(default=len(MultiClass.labels), selection=Columns.label_index)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._calc(df)


class add_label_flag:
    def __init__(self, *labels: str):
        MultiClass.add(*labels)
        self._labels = {*labels}

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for label in MultiClass.labels:
            df.loc[:, (label,)] = label in self._labels
        return df


class map_selection_to_index:
    def __init__(self, *args: str, **kwargs: int):
        self._indices = dict(zip(args, range(len(args))))
        self._indices.update(kwargs)
        self.set()

    def set(
        self,
        default: int = 0,
        selection: str = ...,
        op: Literal["+", "|"] = "+",
    ):
        self._default = default
        self._selection = selection
        self._op = op
        return self

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        t = np.dtype(Columns.index_dtype)
        idx = np.zeros(len(df), dtype=t)
        sel = np.zeros(len(df), dtype=bool)
        for k, v in self._indices.items():
            arr = df[k].to_numpy(dtype=bool)
            match self._op:
                case "+":
                    idx += arr * t.type(v)
                case "|":
                    idx |= arr * t.type(v)
            sel |= arr
        idx[~sel] = t.type(self._default)
        df.loc[
            :, (Columns.selection_index if self._selection is ... else self._selection,)
        ] = idx
        return df


class add_columns:
    def __init__(self, **columns: str):
        self.columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for k, v in self.columns.items():
            df.loc[:, (k,)] = v
        return df


class drop_columns:
    def __init__(self, *columns: str):
        self.columns = [*columns]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=self.columns)


class rename_columns:
    def __init__(self, **columns: str):
        self.columns = columns

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df.rename(columns=self.columns, inplace=True)
        return df


class prescale:
    def __init__(
        self,
        scale: str | Fraction,
        selection: Callable[[pd.DataFrame], npt.ArrayLike] = None,
        seed: SeedLike = (0,),
        additional_columns: Iterable[str] = None,
    ):
        scale = Fraction(scale)
        if scale < 1:
            raise ValueError(f"Scale must be greater than 1, got {scale}")
        self._scale = 1 / scale
        self._selection = selection
        self._columns = set(additional_columns or ())
        self._rng = Squares((type(self).__name__, *seed))

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scale == 1:
            return df
        import numpy as np

        columns = [*(self._columns | {Columns.weight})]
        if self._selection is not None:
            prescaled = np.asarray(self._selection(df))
        else:
            prescaled = np.ones(len(df), dtype=np.bool_)
        unprescaled = ~prescaled
        sumw = df.loc[prescaled, columns].sum(axis=0)
        prescaled[prescaled] = keep_fraction(
            self._scale, self._rng.uint64(df.loc[prescaled, Columns.event])
        )
        sumw_kept = df.loc[prescaled, columns].sum(axis=0)
        df.loc[prescaled, columns] *= sumw / sumw_kept
        return df[prescaled | unprescaled]
