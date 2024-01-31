from typing import Literal

import pandas as pd

# TODO accept train and evaluation
class KFold:
    def __init__(
            self,
            k: int,
            valid: int,
            method: Literal['sequential']= 'sequential'):
        if valid >= k:
            raise ValueError('size of validation set must be smaller than number of folds')
        self.k = k
        self.valid = valid
        self.method = method

    def _sequential(self, index: pd.Series, return_offset: bool):
        ... # TODO yield train, valid, offset

    def split(self, df: pd.DataFrame, return_offset: bool = False):
        ... # TODO yield train, valid, offset
    
    def merge(self):
        ... # TODO return merged df