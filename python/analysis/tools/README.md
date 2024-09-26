# Some useful tools

## Merge friend tree metafiles

```console
python -m analysis.tools.merge_friend_meta [-h] -o OUTPUT -i INPUT [INPUT ...] [--cleanup]

options:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        input metafiles
  -o OUTPUT, --output OUTPUT
                        output metafile
  --cleanup             remove input metafiles after merging
```

## Validate MC datasets

Find the errors and weight outliers in the MC datasets.

```console
python -m analysis.tools.mc_dataset_validation [-h] -m METADATAS [METADATAS ...] [-d DATASETS [DATASETS ...]] [-o OUTPUT] [--threshold THRESHOLD] [--sites SITES [SITES ...]] [--workers WORKERS] [--condor] [--dashboard DASHBOARD]

options:
  -h, --help            show this help message and exit
  -m METADATAS [METADATAS ...], --metadatas METADATAS [METADATAS ...]
                        path to metadata files
  -d DATASETS [DATASETS ...], --datasets DATASETS [DATASETS ...]
                        dataset names, if not provided, all MC datasets will be used
  -o OUTPUT, --output OUTPUT
                        output directory
  --threshold THRESHOLD
                        threshold to determine outliers comparing to the median
  --sites SITES [SITES ...]
                        priority of sites to read root files
  --workers WORKERS     max number of workers when analyzing the events
  --condor              submit analysis jobs to condor
  --dashboard DASHBOARD
                        dashboard address
```
