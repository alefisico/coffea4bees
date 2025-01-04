#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/skimmer_fourTag_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
#time python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/HH4b_fourTag.yml -y UL18 UL17 UL16_preVFP UL16_postVFP -d data -op skimmer/metadata/ -o picoaod_datasets_fourTag_data_Run2.yml -m metadata/datasets_HH4b.yml
time python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/HH4b_fourTag.yml -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix -d data -op ${OUTPUT_DIR} -o picoaod_datasets_fourTag_data_Run3.yml -m metadata/datasets_HH4b_Run3.yml 
#ls -R skimmer/

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi