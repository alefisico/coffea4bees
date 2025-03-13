#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}synthetic_dataset_analyze"
OUTPUT_DIR="${DEFAULT_DIR}synthetic_dataset_analyze_cutflow"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running dump cutflow test"
python analysis/tests/dumpCutFlow.py --input $INPUT_DIR/test_synthetic_datasets.coffea -o $OUTPUT_DIR/test_dump_cutflow_synthetic_datasets.yml
echo "############### Running cutflow test"
python analysis/tests/cutflow_test.py   --inputFile $INPUT_DIR/test_synthetic_datasets.coffea --knownCounts analysis/tests/known_counts_test_synthetic_datasets.yml
ls $OUTPUT_DIR/test_dump_cutflow_synthetic_datasets.yml

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi