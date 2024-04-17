echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/test_systematics.coffea --knownCounts analysis/tests/testCounts_systematics.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input analysis/hists/test_systematics.coffea -o analysis/tests/test_dump_systematics_cutflow.yml
ls analysis/tests/test_systematics_dump_cutflow.yml
cd ../

