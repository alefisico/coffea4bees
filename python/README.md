# Python code

This is a [coffea](https://coffeateam.github.io/coffea/index.html) based analysis. To run the content of this folder one must set the cointainer using singularity described in [here](../README.md). 

_All the code in this folder assumes that this is the root folder, i.e. all the references are based from this directory._

Here the folders contain:
 - [base_class](./base_class/): common files for defining plots, hist, datasets in rucio, etc. All the base classes used in all the analysis.
 - [skimmer](./skimmer/): processors to take nanoAOD files, apply some selection and same skimmed nanoAOD files (aka picoAOD files).
 - [analysis](./analysis/): analysis processor to create histograms with selection and files to make plots
 - [classifier](./classifier/): __add info__
 - [data](./data/): location of correction files (to be replaced by the use of the json-pog)
 
In this folder you can also find the `runner.py`, which is the master runner for all the processors in the following folders. Currently the arguments are:

```
usage: runner.py [-h] [-t] [-o OUTPUT_FILE] [-p PROCESSOR] [-c CONFIGS] [-m METADATA] [-op OUTPUT_PATH]
                 [-y {UL16_postVFP,UL16_preVFP,UL17,UL18} [{UL16_postVFP,UL16_preVFP,UL17,UL18} ...]]
                 [-d DATASETS [DATASETS ...]] [-s] [--condor] [--debug]

Run coffea processor

options:
  -h, --help            show this help message and exit
  -t, --test            Run as a test with few files (default: False)
  -o OUTPUT_FILE, --output OUTPUT_FILE
                        Output file. (default: hists.coffea)
  -p PROCESSOR, --processor PROCESSOR
                        Processor file. (default: analysis/processors/processor_HH4b.py)
  -c CONFIGS, --configs CONFIGS
                        Config file. (default: analysis/metadata/HH4b.yml)
  -m METADATA, --metadata METADATA
                        Metadata datasets file. (default: metadata/datasets_HH4b.yml)
  -op OUTPUT_PATH, --outputPath OUTPUT_PATH
                        Output path, if you want to save file somewhere else. (default: hists/)
  -y {UL16_postVFP,UL16_preVFP,UL17,UL18} [{UL16_postVFP,UL16_preVFP,UL17,UL18} ...], --year {UL16_postVFP,UL16_pre
VFP,UL17,UL18} [{UL16_postVFP,UL16_preVFP,UL17,UL18} ...]
                        Year of data to run. Example if more than one: --year UL17 UL18 (default: ['UL18'])
  -d DATASETS [DATASETS ...], --datasets DATASETS [DATASETS ...]
                        Name of dataset to run. Example if more than one: -d HH4b ZZ4b (default: ['HH4b', 'ZZ4b',
                        'ZH4b'])
  -s, --skimming        Run skimming instead of analysis (default: False)
  --condor              Run in condor (default: False)
  --debug               Print lots of debugging statements (default: False)
```

More information about each process in the README.md of each folder.
