echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running base class test"
python base_class/tests/plots_test.py --inputFile analysis/hists/test.coffea --known base_class/tests/known_PlotCounts.yml
python base_class/tests/dumpPlotCounts.py --input analysis/hists/test.coffea --output base_class/tests/test_dumpPlotCounts.yml
ls base_class/tests/test_dumpPlotCounts.yml
cd ../

