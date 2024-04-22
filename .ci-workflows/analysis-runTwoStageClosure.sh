
# In coffea envi
echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Hist --> YML"
python stats_analysis/convert_hist_to_yaml.py --input analysis/hists/testMixedData_master.coffea --output analysis/hists/testMixedData_master.yml
python stats_analysis/convert_hist_to_yaml.py --input analysis/hists/testMixedBkg_master.coffea --output analysis/hists/testMixedBkg_master.yml 


# In root envirornment
py stats_analysis/convert_yml_to_root.py -f analysis/hists/testMixedData_master.yml --output analysis/hists/
py stats_analysis/convert_yml_to_root.py -f analysis/hists/testMixedBkg_master.yml --output analysis/hists/



