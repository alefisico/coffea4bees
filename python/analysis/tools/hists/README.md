# Hists related tools

## Group hists by category

Group the hists based on category axes and save in separate files.

```console
python -m analysis.tools.hists.group_hists_by_category [-h] -i INPUT -o OUTPUT

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path to the input hist file (default: None)
  -o OUTPUT, --output OUTPUT
                        path to the output hist file (default: None)
```

## Copy data to multijet

Since, the reweighted 3b data are treated as 4b multijet background model, to make it compatible with the plotting tool, the hists under `{process: data, ntag: 3}` need to be copied to `{process: QCD Multijet, ntag: 4}`.

```console
python -m analysis.tools.hists.data_to_multijet [-h] -i INPUT_FILES [INPUT_FILES ...] [-o OUTPUT_PATTERN] [-p data multijet] [--process-axis PROCESS_AXIS] [--tag-axis TAG_AXIS]

options:
  -h, --help            show this help message and exit
  -i INPUT_FILES [INPUT_FILES ...], --input-files INPUT_FILES [INPUT_FILES ...]
                        path to input hist files (default: [])
  -o OUTPUT_PATTERN, --output-pattern OUTPUT_PATTERN
                        output path pattern (see documentation for path_from_pattern) (default: {host}{parent1}/{name}_mj.{ext})
  -p data multijet, --processes data multijet
                        name of data and multijet processes (default: [])
  --process-axis PROCESS_AXIS
                        name of process axis (default: process)
  --tag-axis TAG_AXIS   name of tag axis (default: tag)
```
