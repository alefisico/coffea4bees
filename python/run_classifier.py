import logging
from datetime import datetime

from base_class.addhash import get_git_diff, get_git_revision_hash
from classifier.task import EntryPoint

logging.getLogger().setLevel(logging.INFO)


def reproducible():
    return {
        "date": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "hash": get_git_revision_hash(),
        "diff": get_git_diff(),
    }


if __name__ == "__main__":
    main = EntryPoint()
    main.run(reproducible)
