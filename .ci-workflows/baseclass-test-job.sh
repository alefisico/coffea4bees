#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=false ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}analysis_test_job"
OUTPUT_DIR="${DEFAULT_DIR}baseclass_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
echo "############### Running base class test"
python base_class/tests/dumpPlotCounts.py --input $INPUT_DIR/test.coffea --output $OUTPUT_DIR/test_dumpPlotCounts.yml
python base_class/tests/plots_test.py --inputFile $INPUT_DIR/test.coffea --known base_class/tests/known_PlotCounts.yml
ls $OUTPUT_DIR/test_dumpPlotCounts.yml

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi