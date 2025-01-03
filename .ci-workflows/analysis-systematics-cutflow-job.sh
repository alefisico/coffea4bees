#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=false ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}analysis_systematics_test_job"
OUTPUT_DIR="${DEFAULT_DIR}analysis_systematics_cutflow_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test_systematics.coffea -o $OUTPUT_DIR/test_dump_systematics_cutflow.yml

python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test_systematics.coffea --knownCounts analysis/tests/testCounts_systematics.yml
echo "############### Running dump cutflow test"
ls $OUTPUT_DIR/test_dump_systematics_cutflow.yml

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi