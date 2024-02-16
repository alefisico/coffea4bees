# TODO check
import numpy as np  # TODO move this inside

__all__ = ['Constant']


class Constant:
    event = 'event'
    weight = 'weight'

    index_dtype = np.uint8

    label_index = 'label_index'
    region_index = 'region_index'
    event_offset = 'event_offset'

    unscaled_weight = 'unscaled_weight'

    authkey_size = 512
