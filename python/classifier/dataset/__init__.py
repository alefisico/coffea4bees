from .dataset import MultiTensorDataset, NamedDataLoader
from .label import Label, LabelCollection
from .df_reader import *
from .df_tools import *

__all__ = [
    'MultiTensorDataset',
    'NamedDataLoader',
    'Label',
    'LabelCollection',
    'DF'
]


class DF:
    Reader = Reader

    from_root = FromRoot
    from_friend_tree = FromFriendTree
    to_tensor_dataset = ToTensorDataset

    construct = Constructor
    normalize = Normalizer

    add_unscaled_weight = add_unscaled_weight
    add_region_index = add_region_index
    add_event_offset = add_event_offset
