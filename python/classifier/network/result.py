import pandas as pd

from ..dataset import LabelCollection


class Result:
    def __init__(
            self,
            labels: type[LabelCollection],
            true: pd.DataFrame,
            pred: pd.DataFrame,
            data: pd.DataFrame = None):
        ...  # TODO
