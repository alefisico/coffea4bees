from classifier.df.tools import add_label_flag, add_label_index

from . import ArgParser, Dataframe


class WithSingleLabel(Dataframe):
    argparser = ArgParser()
    argparser.add_argument(
        '--label', required=True, help='the label of the dataset')

    def __init__(self):
        super().__init__()
        self.postprocessors.append(add_label_index(self.opts.label))


class WithMultipleLabel(Dataframe):
    argparser = ArgParser()
    argparser.add_argument(
        '--labels', nargs='+', required=True, help='the labels of the dataset')

    def __init__(self):
        super().__init__()
        self.postprocessors.append(add_label_flag(*self.opts.labels))
