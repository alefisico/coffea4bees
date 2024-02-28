from classifier.process.state import Cascade


class DataLoader(Cascade):
    batch_io: int = 1_000_000
    batch_eval: int = 2**15
    shuffle: bool = True
    num_workers: int = 0


class Model(Cascade):
    kfolds: int = 3
