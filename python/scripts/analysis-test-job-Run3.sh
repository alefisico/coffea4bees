#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy


echo "############### Overwritting datasets"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b_Run3_fourTag_v3.yml
else
    DATASETS=metadata/datasets_HH4b_Run3_cernbox.yml
fi
echo "The datasets file is: $DATASETS"


OUTPUT_DIR="${DEFAULT_DIR}/analysis_test_job_Run3"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
python runner.py -t -o test.coffea -d data  -p analysis/processors/processor_HH4b.py -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/HH4b_run_fastTopReco.yml



ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
