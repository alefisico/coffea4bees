# TODO for test only, will be removed or rewritten
# TODO work in progress to match new skimmed data

from classifier.df.tools import (add_event_offset, add_label_index,
                                 drop_columns, map_selection_to_index)

from ..model.example import features
from ..setting.df import Columns
from ._df import LoadGroupedRoot


class _Basic(LoadGroupedRoot):
    _CanJet = [*map('CanJet_{}'.format, features.candidate_jet)]
    _NotCanJet = [*map('NotCanJet_{}'.format, features.other_jet)]
    _branches = {
        'ZZSR', 'ZHSR', 'HHSR', 'SB',
        'fourTag', 'threeTag', 'passHLT',
        'nSelJets', 'weight', 'event',
        *_CanJet, *_NotCanJet,
    }

    def __init__(self):
        super().__init__()
        (
            self.to_tensor
            .add(Columns.event_offset, Columns.index_dtype).columns(Columns.event_offset)
            .add(Columns.label_index, Columns.index_dtype).columns(Columns.label_index)
            .add('region_index', Columns.index_dtype).columns('region_index')
            .add('weight', 'float32').columns('genWeight')
            .add('ancillary', 'float32').columns(*features.ancillary)
            .add('candidate_jet', 'float32').columns(*self._CanJet, target=features.candidate_jet_max)
            .add('other_jet', 'float32').columns(*self._NotCanJet, target=features.other_jet_max)
        )
        self._preprocessors.extend([
            add_event_offset(60),  # 1, 2, 3, 4, 5, 6 folds
            map_selection_to_index(SB=0b10, ZZSR=0b00101, ZHSR=0b01001, HHSR=0b10001).set(
                selection='region_index'),
            map_selection_to_index(fourTag=0b10, threeTag=0b01).set(
                selection='ntag_index'),
            drop_columns('ZZSR', 'ZHSR', 'HHSR', 'SB',
                         'fourTag', 'threeTag', 'passHLT', 'event'),
        ])

