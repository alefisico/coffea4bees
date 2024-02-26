from classifier.process.state import Cascade


class Dataset(Cascade):
    dataloader_io_batch: int = 1_000_000
    dataloader_shuffle: bool = True
