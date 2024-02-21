__all__ = ['Constant']


class Constant:
    event = 'event'
    weight = 'weight'

    label_index = 'label_index'
    region_index = 'region_index'
    event_offset = 'event_offset'

    unscaled_weight = 'unscaled_weight'

    _index_dtype = ...

    @classmethod
    def index_dtype(cls):
        if cls._index_dtype is ...:
            import numpy as np
            cls._index_dtype = np.uint8
        return cls._index_dtype
