# Some useful tools

## Merge friend tree metafiles

```console
python -m analysis.tools.merge_friend_meta [-h] -i INPUT [INPUT ...] -o OUTPUT [--cleanup]

options:
  -h, --help            show this help message and exit
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        input metafiles (default: [])
  -o OUTPUT, --output OUTPUT
                        output metafile (default: None)
  --cleanup             remove input metafiles after merging (default: False)
```

## Validate MC datasets

Find the errors and weight outliers in the MC datasets.

```console
python -m analysis.tools.mc_dataset_validation [-h] -m METADATAS [METADATAS ...] [-d DATASETS [DATASETS ...]] [-o OUTPUT] [--threshold THRESHOLD]
                                [--sites SITES [SITES ...]] [--workers WORKERS] [--condor] [--dashboard DASHBOARD]

options:
  -h, --help            show this help message and exit
  -m METADATAS [METADATAS ...], --metadatas METADATAS [METADATAS ...]
                        path to metadata files (default: [])
  -d DATASETS [DATASETS ...], --datasets DATASETS [DATASETS ...]
                        dataset names, if not provided, all MC datasets will be used (default: [])
  -o OUTPUT, --output OUTPUT
                        output directory (default: .)
  --threshold THRESHOLD
                        threshold to determine outliers comparing to the median (default: 1000)
  --sites SITES [SITES ...]
                        priority of sites to read root files (default: ['T3_US_FNALLPC'])
  --workers WORKERS     max number of workers when analyzing the events (default: 8)
  --condor              submit analysis jobs to condor (default: False)
  --dashboard DASHBOARD
                        dashboard address (default: :10200)
```
