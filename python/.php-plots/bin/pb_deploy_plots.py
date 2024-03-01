#!/usr/bin/env python3

"""
Copies images recursively to a target directory, adds plot browser index files to all newly created
directories, and optionally converts pdf files to png.
"""

import os
import glob
import shutil
from collections import deque
from typing import Sequence, Union, Optional


this_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(this_dir)


def deploy_plots(
    sources: Union[Sequence[str], str],
    destination: str,
    extensions: Optional[Sequence[str]] = None,
    convert_pdf: bool = False,
    recursive: bool = False,
    n_cores: int = 1,
) -> None:
    if isinstance(sources, str):
        sources = [sources]

    # ensure that the destination directory exists
    destination = os.path.abspath(destination)
    if not os.path.exists(destination):
        os.makedirs(destination)
    elif not os.path.isdir(destination):
        raise RuntimeError(f"{destination} exists but is not a directory")

    # expand patterns
    sources = sum(map(glob.glob, sources), [])

    # convert to (src, dst) pairs
    sources = [
        (os.path.abspath(src), os.path.join(destination, os.path.basename(src)))
        for src in sources
    ]

    # keep track of copied pdf files
    pdf_paths = set()

    # start copying
    print("copying files ...")
    sources = deque(sources)
    seen = set()
    while sources:
        src, dst = sources.popleft()

        # repetition guard
        if src in seen:
            continue
        seen.add(src)

        # handle files
        if os.path.isfile(src):
            if extensions and not dst.endswith(extensions):
                continue
            dst_dir = os.path.dirname(dst)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            shutil.copy2(src, dst)
            if dst.endswith(".pdf"):
                pdf_paths.add(dst)
            continue

        # handle directories
        for elem in os.listdir(src):
            path = os.path.join(src, elem)
            if os.path.isfile(path) or recursive:
                sources.appendleft((path, os.path.join(dst, elem)))
    print("done")

    # copy index files
    from pb_copy_index import copy_index
    copy_index([destination], recursive=True)

    # convert pdf files
    if convert_pdf:
        from pb_pdf_to_png import convert_pdfs
        convert_pdfs(
            [(path, f"{os.path.splitext(path)[0]}.png") for path in pdf_paths],
            n_cores=n_cores,
        )


def main() -> None:
    from argparse import ArgumentParser

    csv = lambda s: tuple(_s.strip() for _s in s.strip().split(",") if _s.strip())

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "sources",
        nargs="+",
        help="source files or directories to check for plots",
    )
    parser.add_argument(
        "destination",
        help="target directory to copy files to",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        type=csv,
        default=(
            "png", "pdf", "jpg", "jpeg", "gif", "eps", "svg", "root", "cxx", "txt", "rtf", "log",
            "csv",
        ),
        help="comma-separated extensions of files to copy; default: %(default)s",
    )
    parser.add_argument(
        "--pdf-to-png",
        "-c",
        action="store_true",
        help="convert pdf files to png",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="convert pdf files recursively in all subdirectories",
    )
    parser.add_argument(
        "--cores",
        "-j",
        type=int,
        default=1,
        help="number of cores to use for parallel conversion of pdf files",
    )
    args = parser.parse_args()

    deploy_plots(
        args.sources,
        args.destination,
        extensions=args.extensions,
        convert_pdf=args.pdf_to_png,
        recursive=args.recursive,
        n_cores=args.cores,
    )


if __name__ == "__main__":
    main()
