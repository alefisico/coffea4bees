# Hists related tools

## Group hists by category

Group the hists based on category axes and save in separate files.

```console
python -m analysis.tools.hists.group_hists_by_category [-h] -f INPUT OUTPUT

options:
  -h, --help            show this help message and exit
  -f INPUT OUTPUT, --hist-files INPUT OUTPUT
                        path to input and output hist files (default: [])
```

## Copy data to multijet

Since, the reweighted 3b data are treated as 4b multijet background model, to make it compatible with the plotting tool, the hists under `{process: data, ntag: 3}` need to be copied to `{process: QCD Multijet, ntag: 4}`.

```console
python -m analysis.tools.hists.data_to_multijet [-h] -f INPUT OUTPUT -p DATA MULTIJET [--process-axis PROCESS_AXIS] [--tag-axis TAG_AXIS]

options:
  -h, --help            show this help message and exit
  -f INPUT OUTPUT, --hist-files INPUT OUTPUT
                        path to input and output hist files (default: [])
  -p DATA MULTIJET, --processes DATA MULTIJET
                        name of data and multijet processes (default: [])
  --process-axis PROCESS_AXIS
                        name of process axis (default: process)
  --tag-axis TAG_AXIS   name of tag axis (default: tag)
```
