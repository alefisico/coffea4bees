#!/usr/bin/env python3

"""
Copies the index.php file of the plot browser to various directories.
"""

import os
import glob
import shutil
from collections import deque
from typing import Sequence, Union


this_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(this_dir)


def copy_index(directories: Union[Sequence[str], str], recursive: bool = False) -> None:
    if isinstance(directories, str):
        directories = [directories]

    # determine the index file
    index_file = os.path.join(repo_dir, "index.php")
    if not os.path.exists(index_file):
        raise RuntimeError(f"index file {index_file} does not exist")
    print("copying index file ...")

    # expand patterns
    directories = sum(map(glob.glob, directories), [])

    # copy into directories
    directories = deque(map(os.path.abspath, directories))
    seen = set()
    n = 0
    while directories:
        directory = directories.popleft()

        # repetition guard
        if directory in seen:
            continue
        seen.add(directory)

        # final check
        if not os.path.isdir(directory):
            raise RuntimeError(f"{directory} is not a directory")

        # copy the file
        shutil.copy2(index_file, os.path.join(directory, "index.php"))
        n += 1

        # extend directories if recursive
        if recursive:
            for elem in os.listdir(directory):
                elem = os.path.join(directory, elem)
                if os.path.isdir(elem):
                    directories.appendleft(elem)

    print(f"done, copied into {n} directories")


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "directories",
        nargs="+",
        help="the directories to copy the index.php file to",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="copy the index.php file recursively into all subdirectories",
    )
    args = parser.parse_args()

    copy_index(args.directories, recursive=args.recursive)


if __name__ == "__main__":
    main()
