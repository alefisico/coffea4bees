from ..dataset import DF, Label, LabelCollection
from ..static import Constant

__all__ = ['Labels']


# TODO read FvT from friend tree, multiply to weight
# TODO add selections
# TODO prescale three-tag data
# keep_fraction = 1/PS
# np.random.seed(seed)
# keep = (data.fourTag) | (np.random.rand(n_selected) < keep_fraction)
# data = data[keep]
# TODO scale up if prescaled
# TODO add year
# TODO add label
# TODO log/dashboard
# log readed file, n event, total weight, negative weight
# TODO flip negative weight data driven background to be positive weight ttbar

# TODO log/dashboard
# log n event, sum weight

# TODO scale up signal weight
# TODO calculate target
# df['target'] = sum([c.index*df[c.abbreviation] for c in classes])

# TODO compute die loss
# fC = torch.FloatTensor([wmj/w, wtt/w, wzz_norm/w, wzh_norm/w, whh_norm/w])
# compute the loss you would get if you only used the class fraction to predict class probability (ie a 4 sided die loaded to land with the right fraction on each class)
# loaded_die_loss = -(fC*fC.log()).sum()

class Labels(LabelCollection):
    mj = Label('Multijet Model')
    tt = Label(R'$t\bar{t}$ MC')
    hh = Label('ggF $HH$ MC')

    bg = Label('Background', mj, tt)
    sg = Label('Signal', hh)


class TrainConfig:
    max_other_jet = 8

    labels = Labels
    reader = DF.from_root(
        ['fourTag', 'threeTag', 'passHLT', 'SR', 'CR', 'SB', Constant.weight,
            Constant.event, 'nSelJets', 'canJet*', 'notCanJet*'],
    ).add_flatten('notCanJet*', max_other_jet, -1)


class EvaluateConfig:
    ...
