echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test for mixed Bkg"
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/testMixedBkg.coffea --knownCounts analysis/tests/testCounts_mixed_bkg.yml
echo "############### Running cutflow test for mixed Data"
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/testMixedData.coffea --knownCounts analysis/tests/testCounts_mixed_data.yml
echo "############### Running dump cutflow test for Bkg and Data"
python analysis/tests/dumpCutFlow.py --input analysis/hists/testMixedData.coffea -o analysis/tests/test_dump_cutflow_mixed_data.yml
python analysis/tests/dumpCutFlow.py --input analysis/hists/testMixedBkg.coffea -o analysis/tests/test_dump_cutflow_mixed_bkg.yml
ls analysis/tests/test_dump_cutflow_mixed_data.yml
cd ../

