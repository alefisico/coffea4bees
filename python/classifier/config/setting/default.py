from classifier.process.state import Cascade


class Dataset(Cascade):
    io_step: int = 1_000_000
    dataloader_shuffle: bool = True
