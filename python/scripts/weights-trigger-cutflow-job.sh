#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}weights_trigger_analysis_job"
OUTPUT_DIR="${DEFAULT_DIR}weights_trigger_cutflow_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test_trigWeight.coffea -o $OUTPUT_DIR/test_dump_cutflow_trigWeight.yml
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test_trigWeight.coffea --knownCounts analysis/tests/known_Counts_trigWeight.yml
ls $OUTPUT_DIR/test_dump_cutflow_trigWeight.yml

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi