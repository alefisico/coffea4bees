echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile analysis/trigger_weights/test_trigWeight.coffea --knownCounts analysis/tests/known_Counts_trigWeight.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input analysis/trigger_weights/test_trigWeight.coffea -o analysis/tests/test_dump_cutflow_trigWeight.yml
ls analysis/tests/test_dump_cutflow_trigWeight.yml
cd ../

