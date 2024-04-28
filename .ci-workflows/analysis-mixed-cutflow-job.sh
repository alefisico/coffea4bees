echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
echo "############### Running cutflow test for mixed "
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/testMixedBkg_TT.coffea --knownCounts analysis/tests/known_Counts_MixedBkg_TT.yml
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/testMixedBkg_data_3b_for_mixed.coffea --knownCounts analysis/tests/known_Counts_MixedBkg_data_3b_for_mixed.yml
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/testMixedData.coffea --knownCounts analysis/tests/known_Counts_MixedData.yml
python analysis/tests/cutflow_test.py   --inputFile analysis/hists/testSignal.coffea --knownCounts analysis/tests/known_Counts_Signal.yml



echo "############### Running dump cutflow test for Bkg and Data"
python analysis/tests/dumpCutFlow.py --input analysis/hists/testMixedBkg_TT.coffea -o analysis/tests/test_dump_MixedBkg_TT.yml
python analysis/tests/dumpCutFlow.py --input analysis/hists/testMixedBkg_data_3b_for_mixed.coffea -o analysis/tests/test_dump_MixedBkg_data_3b_for_mixed.yml
python analysis/tests/dumpCutFlow.py --input analysis/hists/testMixedData.coffea -o analysis/tests/test_dump_MixedData.yml
python analysis/tests/dumpCutFlow.py --input analysis/hists/testSignal.coffea -o analysis/tests/test_dump_Signal.yml
ls analysis/tests/test_dump_MixedBkg_TT.yml
ls analysis/tests/test_dump_MixedBkg_data_3b_for_mixed.yml
ls analysis/tests/test_dump_MixedData.yml
ls analysis/tests/test_dump_Signal.yml
cd ../

