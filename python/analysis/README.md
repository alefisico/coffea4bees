# Coffea4bees analysis

To run the analysis, remember first to set the coffea environment and your grid certificate. If you followed the instructions in the [README.md](../../README.md), the `set_shell.sh` file must be located right after the package, and then:

```{bash}
voms-proxy-init -rfc -voms cms --valid 168:00
cd coffea4bees/ ## Or go to this directory
source set_shell.sh
```

## In this folder

Here you find the code to run the analysis from `picoAODs` and to make some plots. 
Each folder contains:
 - [helpers](./helpers/): python files with funcions/classes generic to the analyses
 - [metadata](./metadata/): yml files with the metadata for each analysis. In these files you can find input files, datasets, cross sections, etc.  
 - [processors](./processors/): python files with the processors for each analysis.
 - [pytorchModels](./pytorchModels/): training models
 - weights: JCM txt files with weights
 - tests: python scripts for testing the code.
 - hists (optional): if you run the `runner.py` without a name of the output folder, this folder will be created to store the pickle files.

Then, the run-all script is called `runner.py` and it is one directory below (in [python/](../../python/)). This script will run local or condor depending on the flag used. To learn all the options of the script, just run:
```
# (inside /coffea4bees/python/)
python runner.py --help
```

## Run Analysis

### Example to run the analysis

For example, to run a processor you can do:
```
#  (inside /coffea4bees/python/)
python runner.py -t -o test.coffea -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu  HH4b  -p analysis/processors/processor_example.py -y UL18  -op analysis/hists/ -m analysis/metadata/example.yml
```

The output file of this process will be `test.coffea` (a coffea output file), which contains many histograms and cutflows. 



## To produce some plots

Assuming that the file with your histograms is called `hists/hists.coffea`, you can run:
```
python analysis/makePlots.py -i hists/hists.coffea  -o testPlotsNew 

```

### To produce some plots interactively

```
python -i analysis/iPlot.py  hists/hists.coffea  -o testPlotsNew
```

### Examples

```
>>> examples()
```

### 1D Examples

```
>>> plot("SvB_MA.ps_hh",doRatio=1, debug=True, region="SR",cut="passPreSel",rebin=1,rlim=[0,2],norm=1)
>>> plot("SvB_MA_ps_zh",cut="passPreSel",region="SB",doRatio=True,debug=True,ylabel="Entries",norm=False,legend=True,rebin=5,yscale='log')
```

### 2D Examples

```
>>> plot2d("quadJet_min_dr.lead_vs_subl_m",process="TTToHadronic",region=sum,cut="passPreSel")
>>> plot2d("quadJet_min_dr.lead_vs_subl_m",process="TTToHadronic",region=sum,cut="passPreSel",full=3)
```

### To plot the same process from two different cuts

```
>>> plot("canJet0.pt", region="SR", cut=["passSvB","failSvB"],process="data")
>>> plot("canJet0.pt", region=["SB","SR"],cut="passSvB",process="data")

```

### To plot different processes 

```
>>> plot("v4j.mass", region="SR", cut="passPreSel",process="data",norm=1)
>>> plot("v4j.mass", region="SR", cut="passPreSel",process=["TTTo2L2Nu","data"],norm=1)

```


### To plot the same process from two different inputs

```
> py  -i analysis/iPlot.py hists/histAll_file1.coffea hists/histAll_file1.coffea -l file1 file2
```

```
>>> plot("canJet0.pt",region="SR",cut="passPreSel",process="data")
```



## To debug the code

If you want to debug small portions of the code, you can run it interactively in python by using some commands like:
```{python}
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
fname = "root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18NanoAODv9/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/NANOAODSIM/20UL18JMENano_106X_upgrade2018_realistic_v16_L1v1-v1/230000/9EEE27FD-7337-424F-9D7C-A5427A991D07.root"   #### or any nanoaod file
events = NanoEventsFactory.from_root( fname, schemaclass=NanoAODSchema.v6).events()
```


