#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

INPUT_DIR="${DEFAULT_DIR}analysis_test_job"
OUTPUT_DIR="${DEFAULT_DIR}analysis_makeweights_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running makeweights test"
python analysis/make_weights.py -o $OUTPUT_DIR/testJCM_ROOT   -c passPreSel -r SB --ROOTInputs --i analysis/tests/HistsFromROOTFile.coffea
python analysis/make_weights.py -o $OUTPUT_DIR/testJCM_Coffea -c passPreSel -r SB -i $INPUT_DIR/test.coffea
python analysis/tests/make_weights_test.py

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi