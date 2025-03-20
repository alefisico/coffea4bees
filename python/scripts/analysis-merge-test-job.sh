#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/analysis_merge_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Merging coffea files"
python analysis/tools/merge_coffea_files.py -f $DEFAULT_DIR/analysis_test_job/test_databkgs.coffea $DEFAULT_DIR/analysis_signals_test_job/test_signal.coffea  -o $OUTPUT_DIR/test.coffea

ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
