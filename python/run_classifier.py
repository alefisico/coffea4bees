from datetime import datetime

from base_class.addhash import get_git_diff, get_git_revision_hash
from classifier.monitor.logging import setup_logger
from classifier.patch import patch_awkward_pandas
from classifier.task import EntryPoint


def reproducible():
    return {
        "date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "hash": get_git_revision_hash(),
        "diff": get_git_diff(),
    }


if __name__ == "__main__":
    # setup logging
    setup_logger()
    # install patch
    patch_awkward_pandas()
    # run main
    main = EntryPoint()
    main.run(reproducible)
