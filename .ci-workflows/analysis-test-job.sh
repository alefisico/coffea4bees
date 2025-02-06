#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/analysis_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
python runner.py -t -o test_databkgs.coffea -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu ggZH4b ZH4b ZZ4b -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op $OUTPUT_DIR -m $DATASETS

python runner.py -t -o test_signal.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op $OUTPUT_DIR -m metadata/datasets_HH4b_v1p1.yml -c analysis/metadata/HH4b_signals.yml

python analysis/tools/merge_coffea_files.py -f $OUTPUT_DIR/test_databkgs.coffea $OUTPUT_DIR/test_signal.coffea  -o $OUTPUT_DIR/test.coffea

ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
