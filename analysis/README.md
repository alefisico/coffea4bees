# Coffea4bees analysis

To run the analysis, remember first to set the coffea environment and your grid certificate. If you are on this folder:
```
source ../set_env.sh
voms-proxy-init -rfc -voms cms --valid 168:00
```

## To run the analysis

Then, to run a local job:
```
python coffea_analysis.py -t     
```
for convenience this command is stored in `runTestJob.sh`, you can just run it with `source runTestJob.sh`.


## To produce some plots

Assuming that the file with your histograms is called `test.pkl`, you can run:
```
python uproot_plots.py -i test.pkl
```
