echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
INPUT_DIR="output/analysis_test_mixed_job"
OUTPUT_DIR="output/analysis_mixed_cutflow_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running cutflow test for mixed "
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/testMixedBkg_TT.coffea --knownCounts analysis/tests/known_Counts_MixedBkg_TT.yml
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/testMixedBkg_data_3b_for_mixed.coffea --knownCounts analysis/tests/known_Counts_MixedBkg_data_3b_for_mixed.yml
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/testMixedData.coffea --knownCounts analysis/tests/known_Counts_MixedData.yml
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/testSignal_UL.coffea --knownCounts analysis/tests/known_Counts_Signal.yml


echo "############### Running dump cutflow test for Bkg and Data"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/testMixedBkg_TT.coffea -o $OUTPUT_DIR/test_dump_MixedBkg_TT.yml
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/testMixedBkg_data_3b_for_mixed.coffea -o $OUTPUT_DIR/test_dump_MixedBkg_data_3b_for_mixed.yml
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/testMixedData.coffea -o $OUTPUT_DIR/test_dump_MixedData.yml
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/testSignal_UL.coffea -o $OUTPUT_DIR/test_dump_Signal.yml
ls $OUTPUT_DIR/test_dump_MixedBkg_TT.yml
ls $OUTPUT_DIR/test_dump_MixedBkg_data_3b_for_mixed.yml
ls $OUTPUT_DIR/test_dump_MixedData.yml
ls $OUTPUT_DIR/test_dump_Signal.yml
cd ../

