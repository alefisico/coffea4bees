echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/test.coffea --knownCounts analysis/tests/testCounts.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input analysis/hists/test.coffea -o analysis/tests/test_dump_cutflow.yml
ls analysis/tests/test_dump_cutflow.yml
cd ../

