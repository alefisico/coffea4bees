echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
INPUT_DIR="output/analysis_test_job"
OUTPUT_DIR="output/analysis_cutflow_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test.coffea -o $OUTPUT_DIR/test_dump_cutflow.yml


echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test.coffea --knownCounts analysis/tests/testCounts.yml

ls $OUTPUT_DIR/test_dump_cutflow.yml
cd ../

