echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/test_synthetic_datasets.coffea --knownCounts analysis/tests/known_counts_test_synthetic_datasets.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input analysis/hists/test_synthetic_datasets.coffea -o analysis/tests/test_dump_cutflow_synthetic_datasets.yml
ls analysis/tests/test_dump_cutflow_synthetic_datasets.yml
cd ../

