from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Callable, Iterable, Literal

from ..config.setting.df import Columns
from ..config.state.label import MultiClass

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


class add_event_offset:
    """
    a workaround for no ``uint64`` support in :class:`torch.Tensor`
    """

    def __init__(self, modulus: int):
        self._modulus = modulus

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, (Columns.event_offset,)] = (df[Columns.event] % self._modulus).astype(
            Columns.index_dtype
        )
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
        scale: float,
        selection: Callable[[pd.DataFrame], npt.ArrayLike] = None,
        shuffle: bool = False,
        scale_columns: Iterable[str] = None,
    ):
        if scale < 1:
            raise ValueError(f"Scale must be greater than 1, got {scale}")
        self._scale = 1.0 / scale
        self._selection = selection
        self._shuffle = shuffle
        self._scale_columns = set(scale_columns or ())
        self._scale_columns.add(Columns.weight)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._scale == 1.0:
            return df

        import numpy as np
        import pandas as pd

        if self._selection is not None:
            mask = np.asarray(self._selection(df))
        else:
            mask = np.full(len(df), True)
        index = df.index[mask]
        if self._shuffle:
            np.random.shuffle(index)

        weight = np.abs(df.loc[index, Columns.weight].to_numpy())
        summed = np.cumsum(weight)
        last = np.searchsorted(summed, self._scale * summed[-1])
        scale = summed[last] / summed[-1]
        df.loc[index[: last + 1], [*self._scale_columns]] /= scale

        kept = pd.Series(False, index=df.index)
        kept.loc[~mask] = True
        kept.loc[index[: last + 1]] = True
        return df[kept]
