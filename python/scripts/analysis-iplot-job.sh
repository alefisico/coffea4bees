#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}analysis_test_job"

echo "############### Running iPlot test"
python plots/tests/iPlot_test.py --inputFile $INPUT_DIR/test.coffea

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi