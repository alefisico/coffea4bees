# TODO SvB, FvT, DvT3, DvT4
# TODO for test only, will be removed or rewritten
# TODO work in progress, match new skimmed data

class features:
    canJet = ['pt', 'eta', 'phi', 'm']
    notCanJet = canJet + ['isSelJet']
    ancillary = ['nSelJets']
    ancillary += ['year']
    # ancillary += ['xW', 'xbW']

# .add_column(*(
#     f'canJet{i}_{k}'
#     for k in self.canJet
#     for i in range(4)
# ), dtype=np.float32, name='canJet')
# .add_column(*(
#     f'notCanJet{i}_{k}'
#     for k in self.notCanJet
#     for i in range(self.max_other_jets)
# ), dtype=np.float32, name='notCanJet')
# .add_column(*self.ancillary, dtype=np.float32, name='ancillary')
# .add_column(Constant.label_index, dtype=np.uint8)
# .add_column(Constant.weight, dtype=np.float32)
# .add_column(Constant.region_index, dtype=np.uint8)
# .add_column(Constant.event_offset, dtype=np.uint8)
