#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}analysis_test_job_unsup"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
sed -e "s/condor_cores.*/condor_cores: 6/" -e "s/condor_memory.*/condor_memory: 8GB/" analysis/metadata/unsup4b.yml > $OUTPUT_DIR/unsup4b.yml

cat $OUTPUT_DIR/unsup4b.yml

echo "############### Running test processor"
python runner.py -t -o test_unsup.coffea -d mixeddata data_3b_for_mixed TTToHadronic TTToSemiLeptonic TTTo2L2Nu -p analysis/processors/processor_unsup.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op $OUTPUT_DIR -m $OUTPUT_DIR/unsup4b.yml -c $OUTPUT_DIR/unsup4b.yml
ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi