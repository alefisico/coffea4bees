#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}analysis_test_job_thuthStudy"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
python runner.py -t    -o testTruth.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_genmatch_HH4b.py -y UL18  -op $OUTPUT_DIR -m $DATASETS  -c analysis/metadata/HH4b_genmatch.yml

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi