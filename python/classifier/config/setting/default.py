from __future__ import annotations

from typing import TYPE_CHECKING

from classifier.task import Cascade

if TYPE_CHECKING:
    from base_class.system.eos import EOS


class IO(Cascade):
    output: EOS = "."

    @classmethod
    def _output(cls, var):
        from base_class.system.eos import EOS

        return EOS(var).mkdir(recursive=True)


class DataLoader(Cascade):
    batch_io: int = 1_000_000
    batch_eval: int = 2**15
    shuffle_train: bool = True
    num_workers: int = 0


class Model(Cascade):
    kfolds: int = 3
