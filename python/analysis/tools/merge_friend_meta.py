import json
from argparse import ArgumentParser

import fsspec
from base_class.root.chain import Friend
from base_class.system.eos import EOS, PathLike
from base_class.utils.json import DefaultEncoder


def merge_friend_metas(output: PathLike, *metafiles: PathLike, cleanup: bool = True):
    merged: dict[str, Friend] = {}
    for metafile in metafiles:
        with fsspec.open(metafile) as f:
            meta: dict[str, dict] = json.load(f)
        for k, v in meta.items():
            friend = Friend.from_json(v)
            if k in merged:
                merged[k] += friend
            else:
                merged[k] = friend
    if cleanup:
        for metafile in metafiles:
            EOS(metafile).rm()
    with fsspec.open(EOS(output).mkdir(True), "w") as f:
        json.dump(merged, f, cls=DefaultEncoder)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-o", "--output", required=True, help="Output metafile")
    argparser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="Input metafiles",
        action="extend",
        default=[],
    )
    argparser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove input metafiles after merging",
    )
    args = argparser.parse_args()
    merge_friend_metas(args.output, *args.input, cleanup=args.cleanup)
