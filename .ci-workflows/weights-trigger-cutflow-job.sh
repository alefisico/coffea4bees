echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
INPUT_DIR="output/weights_trigger_analysis_job"
OUTPUT_DIR="output/weights_trigger_cutflow_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test_trigWeight.coffea --knownCounts analysis/tests/known_Counts_trigWeight.yml
echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test_trigWeight.coffea -o $OUTPUT_DIR/test_dump_cutflow_trigWeight.yml
ls $OUTPUT_DIR/test_dump_cutflow_trigWeight.yml
cd ../

