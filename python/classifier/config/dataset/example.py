# TODO for test only, will be removed or rewritten
# TODO work in progress to match new skimmed data

from classifier.df.tools import (add_event_offset, map_selection_to_index,
                                 add_label_index, normalize_weight)

from ._df import LoadGroupedRoot


def sel(df):
    # TODO
    return df


class all(LoadGroupedRoot):
    def __init__(self):
        super().__init__()
        (self.to_tensor
         .add('event', 'int64').columns('event', 'run')
         .add('canJet', '').columns('canJet_pt', 'canJet_eta', target=4)
         .add('label').columns('label_index'))
        self._postprocessors.extend(
            [add_event_offset(3), normalize_weight(100), map_selection_to_index('SR', 'CR').set(default=5)])

    @property
    def from_root(self):
        from classifier.df.io import FromRoot
        return {
            'd3': FromRoot(
                friends=self.friends,
                branches={'Jet_pt', 'Jet_eta',
                          'event', 'run', 'genWeight'}.intersection,
                preprocessors=[sel, add_label_index(
                    'bb')] + self.preprocessors,
                metadata={'year': 18},
            ),
        }
