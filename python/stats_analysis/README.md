# Coffea4bees statistical analysis

This part of the analysis uses [coffea](https://coffeateam.github.io/coffea/), [ROOT](https://root.cern/) and the cms package [combine](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/). Therefore we need different environments to run the different steps.

We need two sets of environemnts. One with `coffea` to take the outputs of analysis and convert them into a json format and another with `root` and `combine`. For the first one, you can use the `coffea4bees` container. For the second follow the next instructions.

## Install Combine and CombineHarvester

### With container

We can use the combine/combineHarvester container. For that you can run:
```
cd coffea4bees/  ### if you are not there
./shell_combine
```
And after that you **must** run the following script inside the container:
```
cd coffea4bees/python/stats_analysis/   #### if you are not there
source set_cmssw.sh
```

### In sl7 machines


For the second part ( with `root` and `combine`), we will set a `CMSSW` environment. You can run the next lines outside your `coffea4bees` directory if you want to. 

```
cmsrel CMSSW_11_3_4
cd CMSSW_11_3_4/src
cmsenv
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
git fetch origin
git checkout v9.2.0
scram b -j 5
cd $CMSSW_BASE/src/
bash <(curl -s https://raw.githubusercontent.com/cms-analysis/CombineHarvester/main/CombineTools/scripts/sparse-checkout-https.sh)
scram b -j 5
cmsenv
```

This steps follow the environment recommended for the combine team,  [here](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/#combine-v9-recommended-version)).
This setup you have to do it only once, however **you need to set this environment anytime you want to use root or combine**.


## Convert hist to json

Using the coffea4bees container:
```
cd python/stats_analysis/
python convert_hist_to_json.py -o histos/histAll.json -i ../analysis/hists/histAll.coffea
```

## Convert json to root (for combine)

Using the CMSSW environment described before:
```
cd $CMSSW_BASE/src/                       ##### directory where CMSSW is located
cmsenv
cd ~/coffea4bees/python/stats_analysis/   ##### directory where coffea4bees is located
python convert_json_to_root.py --classifier SvB_MA SvB -f histos/histAll.json --merge2016 --output_dir datacards/ --plot
```
