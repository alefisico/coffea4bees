# Python code

## Structure of the Code

This project is built using [coffea](https://coffeateam.github.io/coffea/index.html), a framework for high-energy physics analyses.

_All code in this directory assumes it is the root folder, meaning all file paths and references are relative to this directory._

### Folder Overview

- **[base_class](./base_class/):** Contains common utilities and base classes for defining plots, histograms, datasets in Rucio, and other shared functionality used across the analysis.
- **[skimmer](./skimmer/):** Includes processors for filtering NanoAOD files, applying selections, and saving skimmed NanoAOD files (referred to as picoAOD files).
- **[analysis](./analysis/):** Houses the analysis processor for creating histograms with selections and scripts for generating plots.
- **[classifier](./classifier/):** Contains machine learning models and scripts for classification tasks. _(Details to be added.)_
- **[data](./data/):** Stores correction files, which will eventually be replaced by the use of JSON-based POG files.
- **[jet_clustering](./jet_clustering/):** Includes tools for jet clustering and synthetic data generation.
- **[metadata](./metadata/):** Contains datasets, cross-sections, friend trees, and other metadata files.
- **[plots](./plots/):** Includes all scripts for generating visualizations and plots.
- **[stats_analysis](./stats_analysis/):** Contains scripts for statistical analysis using the Combine framework.
- **[workflows](./workflows/):** Defines Snakemake workflows for automating various tasks.

For more details about each component, refer to the `README.md` file located in the respective folder.

## How to run the container for the code

The project utilizes three distinct containers, each tailored for specific tasks:

- **Analysis Container**: A coffea-based container designed for tasks such as skimming, analysis, jet clustering, and other operations related to reading NanoAOD/picoAOD files, analyzing them, and generating histograms.
- **Combine Container**: The official CMS Combine container, used exclusively for statistical analysis tasks.
- **Snakemake Container**: A snakemake-based container for running workflows and automating processes.

The required software for this package can be executed either interactively or within the containers using the [run_container](./run_container) script. 

- **Interactive Mode**: The container remains open until you manually close it, allowing you to execute commands interactively.
- **Job Mode**: The container is launched temporarily to execute a specific job and closes automatically once the job is completed.

```bash
Usage: ./run_container [command] [options]

Commands:
   [command...]         Run commands inside the analysis container.
                        Opens an interactive shell if no commands are given.
                        (Interactive shell is the only option to run on LPC HTCondor).
  combine [command...]  Run commands inside the combine container.
                        Opens an interactive shell if no commands are given.
  snakemake [options]   Run snakemake with the specified options.
                        Requires --snakefile argument.
  --help                Show this help message.

Examples:
  source run_container                Open an interactive shell in the analysis container. 
                                      This is the only option for HTCondor jobs.
  ./run_container                     Open an interactive shell in the analysis container.
  ./run_container python --version    Run 'python --version' in the analysis container.
  ./run_container combine             Open an interactive shell in the combine container.
  ./run_container combine combine -M AsymptoticLimits  Run combine in the combine container.
  ./run_container snakemake --snakefile Snakefile  Run snakemake with the specified Snakefile.
```

## How to run the coffea part of the code

To execute the analysis portion of the code, use the `runner.py` script, which operates within the analysis container (refer to the container details above). This script serves as the primary entry point for running all processors located in the specified folders. Below is an overview of its arguments:

```bash
usage: runner.py [-h] [-t] [-o OUTPUT_FILE] [-p PROCESSOR] [-c CONFIGS] [-m METADATA] [-op OUTPUT_PATH]
                 [-y {2016,2017,2018,UL16_postVFP,UL16_preVFP,UL17,UL18,2022_preEE,2022_EE,2023_preBPix,2023_BPix} [{2016,2017,2018,UL16_postVFP,UL16_preVFP,UL17,UL18,2022_preEE,2022_EE,2023_preBPix,2023_BPix} ...]]
                 [-d DATASETS [DATASETS ...]] [-e ERA [ERA ...]] [--systematics] [-s] [--dask] [--condor] [--debug] [--githash GITHASH] [--gitdiff GITDIFF] [--check_input_files]

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
  -y {2016,2017,2018,UL16_postVFP,UL16_preVFP,UL17,UL18,2022_preEE,2022_EE,2023_preBPix,2023_BPix} [{2016,2017,2018,UL16_postVFP,UL16_preVFP,UL17,UL18,2022_preEE,2022_EE,2023_preBPix,2023_BPix} ...], --year {2016,2017,2018,UL16_postVFP,UL16_preVFP,UL17,UL18,2022_preEE,2022_EE,2023_preBPix,2023_BPix} [{2016,2017,2018,UL16_postVFP,UL16_preVFP,UL17,UL18,2022_preEE,2022_EE,2023_preBPix,2023_BPix} ...]
                        Year of data to run. Example if more than one: --year UL17 UL18 (default: ['UL18'])
  -d DATASETS [DATASETS ...], --datasets DATASETS [DATASETS ...]
                        Name of dataset to run. Example if more than one: -d HH4b ZZ4b (default: ['HH4b', 'ZZ4b', 'ZH4b'])
  -e ERA [ERA ...], --era ERA [ERA ...]
                        For data only. To run only on one data era. (default: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
  --systematics         Run Systematics for analysis processor (default: False)
  -s, --skimming        Run skimming instead of analysis (default: False)
  --dask                Run with dask (default: False)
  --condor              Run in condor (default: False)
  --debug               Print lots of debugging statements (default: False)
  --githash GITHASH     Overwrite git hash for reproducible (default: )
  --gitdiff GITDIFF     Overwrite git diff for reproducible (default: )
  --check_input_files   Check input files for corruption (default: False)
```
