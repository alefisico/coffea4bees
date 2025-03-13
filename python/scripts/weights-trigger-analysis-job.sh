#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

INPUT_DIR="${DEFAULT_DIR}weights_trigger_friendtree_job"
OUTPUT_DIR="${DEFAULT_DIR}weights_trigger_analysis_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s|trigWeight: .*|trigWeight: \/srv\/python\/$INPUT_DIR\/trigger_weights_friends.json@@trigWeight|" analysis/metadata/HH4b_signals.yml > $OUTPUT_DIR/trigger_weights_HH4b.yml
else
    sed -e "s|trigWeight: .*|trigWeight: \/builds\/${CI_PROJECT_PATH}\/python\/$INPUT_DIR\/trigger_weights_friends.json@@trigWeight|" analysis/metadata/HH4b_signals.yml > $OUTPUT_DIR/trigger_weights_HH4b.yml
fi
cat $OUTPUT_DIR/trigger_weights_HH4b.yml

echo "############### Running test processor"
python runner.py -t -o test_trigWeight.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $OUTPUT_DIR/trigger_weights_HH4b.yml -m $DATASETS
ls -lR ${OUTPUT_DIR}

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi