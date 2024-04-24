
# In coffea envi
echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
#echo "############### Hist --> YML"
#python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/testMixedData_master.coffea  --output analysis/hists/testMixedData_master.yml
#python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/testMixedBkg_master.coffea   --output analysis/hists/testMixedBkg_master.yml 
#python stats_analysis/convert_hist_to_yaml_closure.py --input analysis/hists/histAll.coffea               --output analysis/hists/histAll_signal.yml
#
#
## In root envirornment
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/testMixedData_master.yml --output analysis/hists/
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/testMixedBkg_master.yml --output analysis/hists/
#python3 stats_analysis/convert_yml_to_root.py -f analysis/hists/histAll_signal.yml --output analysis/hists/


# runTwoStage closure
time python3 stats_analysis/runTwoStageClosure.py --channels hh  --classifier SvB_MA
#python3 stats_analysis/tests/dumpTwoStageInputs.py --input stats_analysis/hists_closure_3bDvTMix4bDvT_New.root --output stats_analysis/tests/twoStageClosureInputsCounts.yml
python3 stats_analysis/tests/test_runTwoStageClosure.py --knownCounts stats_analysis/tests/twoStageClosureInputsCounts.yml --input stats_analysis/hists_closure_3bDvTMix4bDvT_New.root 
