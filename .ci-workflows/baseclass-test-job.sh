echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running base class test"
python base_class/tests/plots_test.py --inputFile analysis/hists/test.coffea --known base_class/tests/testPlotCounts.yml 
cd ../

