import numpy as np  # TODO move this inside

from dataclasses import dataclass

__all__ = ['Constant', 'Setting']


@dataclass  # TODO
class Address:
    address: tuple[str, int]
    pipe: str


class Constant:
    event = 'event'
    weight = 'weight'

    index_dtype = np.uint8

    label_index = 'label_index'
    region_index = 'region_index'
    event_offset = 'event_offset'

    unscaled_weight = 'unscaled_weight'

    authkey_size = 512


class Setting:
    debug: bool = True

    console: bool = True
    dashboard: bool = True

    use_cuda: bool = True

    monitor_address: tuple[str, int] = ('localhost', 6000)
    monitor_pipe: str = R'\\.\pipe\monitor_listener'
