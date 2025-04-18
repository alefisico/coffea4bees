import json
from argparse import ArgumentParser

import fsspec
from base_class.root import Friend
from base_class.utils.argparser import DefaultFormatter
from base_class.utils.json import DefaultEncoder
from classifier.task import parse

if __name__ == "__main__":
    argparser = ArgumentParser(formatter_class=DefaultFormatter)
    argparser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="input metafiles",
        action="extend",
        default=[],
    )
    argparser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output metafile",
    )
    args = argparser.parse_args()

    merged: Friend = None
    for path in args.input:
        friend = Friend.from_json(parse.mapping(path, "file"))
        if merged is None:
            merged = friend
        else:
            merged += friend

    if merged is not None:
        with fsspec.open(args.output, "wt") as f:
            json.dump(merged, f, cls=DefaultEncoder)
