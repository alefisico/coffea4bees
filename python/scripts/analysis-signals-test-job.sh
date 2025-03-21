#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/analysis_signals_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
sed -e "s|hist_cuts: .*|hist_cuts: [ passPreSel, passSvB, failSvB ]|" analysis/metadata/HH4b_signals.yml > $OUTPUT_DIR/HH4b_signals.yml
cat $OUTPUT_DIR/HH4b_signals.yml

echo "############### Running test processor"
python runner.py -t -o test_signal.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op $OUTPUT_DIR -m metadata/datasets_HH4b_v1p1.yml -c $OUTPUT_DIR/HH4b_signals.yml

ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
