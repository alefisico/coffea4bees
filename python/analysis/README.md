# Coffea4bees analysis

To run the analysis, remember first to set the coffea environment and your grid certificate. If you followed the instructions in the [README.md](../../README.md), the `shell` file must be located right after the 
```
voms-proxy-init -rfc -voms cms --valid 168:00
source ../shell
```

## In this folder

Here you find the code to run the analysis from `picoAODs` and to make some plots. (The skimming part is still not here, Nov 2023) 
Each folder contains:
 - data: files needed for corrections (to be eliminated in favor or jsonpog)
 - helpers: python files with funcions/classes generic to the analyses
 - metadata: yml files with the metadata for each analysis. In these files you can find input files, datasets, cross sections, etc. 
 - processors: python files with the processors for each analysis.
 - pytorchModels: training models
 - weights: JCM txt files with weights
 - hists (optional): if you run the `runner.py` without a name of the output folder, this folder will be created to store the pickle files.

Then, the run-all script is called `runner.py` and it is one directory below (in [analysis/](../analysis/)). This script will run local or condor depending on the flag used. To learn all the options of the script, just run:
```
# (inside /coffea4bees/python/)
python run.py --help
```

## To run the analysis

For example, to run a processor you can do:
```
#  (inside /coffea4bees/python/)
python runner.py -t -o test.coffea -d HH4b mixeddata -op analysis/hists/ -p analysis/processors/processor_HH4b.py
```

The output file of this process will be `test.coffea` (a coffea output file), which contains many histograms and cutflows. 

For convenience this command is stored in `runTestJob.sh`, you can just run it with `source runTestJob.sh`.


## To produce some plots

Assuming that the file with your histograms is called `hists/hists.coffea`, you can run:
```
python analysis/makePlots.py -i hists/hists.coffea  -o testPlotsNew 

```

## To produce some plots interactively

```
py -i analysis/iPlot.py      -i hists/hists.coffea  -o testPlotsNew

>>> plot("SvB_MA_ps_zh",cut="passPreSel",region="SB",doRatio=True,debug=True,ylabel="Entries",norm=False,legend=True,rebin=5,yscale='log')

```