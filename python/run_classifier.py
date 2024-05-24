from classifier.patch import patch_awkward_pandas
from classifier.task import EntryPoint

if __name__ == "__main__":
    # install patch
    patch_awkward_pandas()
    # run main
    main = EntryPoint()
    main.run()
