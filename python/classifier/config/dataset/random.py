# TODO generate more complex random datasets
from typing import Iterable, TypedDict

from classifier.task import ArgParser, Dataset, converter, parse

from ..state.label import MultiClass


class SimpleUniform(Dataset):
    argparser = ArgParser(description="[red]Work in Progress[/red]")
    argparser.add_argument(
        "--size",
        type=converter.int_pos,
        required=True,
        help="the size of the random dataset",
    )
    argparser.add_argument(
        "--shape",
        type=parse.mapping,
        required=True,
        help=f"the shape of the random data {parse.EMBED}",
    )
    argparser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="the labels of the random dataset",
    )

    def train(self):
        MultiClass.add(*self.opts.labels)
        return [_generate_simple_uniform(self.opts.size, self.opts.shape)]


class _SimpleUniformShape(TypedDict):
    shape: int | Iterable[int]
    dtype: str


class _generate_simple_uniform:
    def __init__(self, size: int, shape: dict[str, _SimpleUniformShape]):
        self.size = size
        self.shape = shape

    def __call__(self):
        import numpy as np
        import torch

        dataset = {}
        for k, v in self.shape.items():
            shape = v["shape"]
            if isinstance(shape, Iterable):
                shape = (self.size, *shape)
            else:
                shape = (self.size, shape)
            dataset[k] = torch.from_numpy(
                np.random.uniform(size=shape).astype(v["dtype"])
            )
        return dataset
