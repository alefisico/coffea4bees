#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/sub_sample_dataset_make_dataset_all"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/sub_sample_MC.py -c skimmer/metadata/sub_sampling_MC.yml -y UL17 UL18 UL16_preVFP UL16_postVFP  -d TTToHadronic TTToSemiLeptonic TTTo2L2Nu -op ${OUTPUT_DIR} -o picoaod_datasets_TT_pseudodata_Run2.yml -m metadata/datasets_HH4b.yml
ls -R ${OUTPUT_DIR}

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi