#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}analysis_test_job_Run3"
OUTPUT_DIR="${DEFAULT_DIR}analysis_cutflow_job_Run3"

echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test.coffea -o $OUTPUT_DIR/test_dump_cutflow.yml


echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test.coffea --knownCounts analysis/tests/testCountsRun3.yml

ls $OUTPUT_DIR/test_dump_cutflow.yml

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
