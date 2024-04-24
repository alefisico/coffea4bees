echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running iPlot test"
python analysis/tests/iPlot_test.py --inputFile analysis/hists/test.coffea
cd ../
