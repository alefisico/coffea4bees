#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/weights_trigger_friendtree_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#make_.*#make_classifier_input: \/srv\/$OUTPUT_DIR\/#" analysis/metadata/trigger_weights.yml > $OUTPUT_DIR/trigger_weights.yml
else
    sed -e "s#make_.*#make_classifier_input: \/builds\/${CI_PROJECT_PATH}\/$OUTPUT_DIR\/#" analysis/metadata/trigger_weights.yml > $OUTPUT_DIR/trigger_weights.yml
fi
cat $OUTPUT_DIR/trigger_weights.yml

echo "############### Running test processor"
python runner.py -t -o trigger_weights_friends.json -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_trigger_weights.py -y UL18 -op $OUTPUT_DIR  -c $OUTPUT_DIR/trigger_weights.yml -m $DATASETS
ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi