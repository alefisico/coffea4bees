# Coffea4bees analysis

To run the analysis, remember first to set the coffea environment and your grid certificate. If you are on this folder:
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
 - hists (optional): if you run the `coffea_analysis.py` without a name of the output folder, this folder will be created to store the pickle files.

Then, the run-all script is called `coffea_analysis.py`. This script will run local or condor depending on the flag used. To learn all the options of the script, just run:
```
python coffea_analysis.py --help
```

## To run the analysis

Then, to run a local job:
```
python coffea_analysis.py -t     
```
for convenience this command is stored in `runTestJob.sh`, you can just run it with `source runTestJob.sh`.


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