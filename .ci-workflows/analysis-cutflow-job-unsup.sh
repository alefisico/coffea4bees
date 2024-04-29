echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/test_unsup.coffea --knownCounts analysis/tests/testCounts_unsup.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input analysis/hists/test_unsup.coffea -o analysis/tests/test_dump_cutflow_unsup.yml
ls analysis/tests/test_dump_cutflow_unsup.yml
cd ../