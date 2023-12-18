echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile hists/test.coffea --knownCounts analysis/tests/testCounts.yml
cd ../

