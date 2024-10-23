echo "############### Checking ls"
ls
echo "############### Moving to python folder"
cd python/
INPUT_DIR="output/analysis_systematics_test_job"
OUTPUT_DIR="output/analysis_systematics_cutflow_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test_systematics.coffea -o $OUTPUT_DIR/test_dump_systematics_cutflow.yml

python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test_systematics.coffea --knownCounts analysis/tests/testCounts_systematics.yml
echo "############### Running dump cutflow test"
ls $OUTPUT_DIR/test_dump_systematics_cutflow.yml
cd ../

