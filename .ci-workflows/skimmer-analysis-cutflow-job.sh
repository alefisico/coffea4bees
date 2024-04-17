echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/test_skimmer_ci.coffea --knownCounts analysis/tests/testCounts_skimmer.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input analysis/hists/test_skimmer_ci.coffea -o analysis/tests/test_dump_skimmer_cutflow.yml
ls analysis/tests/test_dump_skimmer_cutflow.yml
cd ../

