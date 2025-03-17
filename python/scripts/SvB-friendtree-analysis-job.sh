#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

INPUT_DIR="${DEFAULT_DIR}SvB_friendtree_job"
OUTPUT_DIR="${DEFAULT_DIR}SvB_friendtree_analysis_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s|SvB: .*|SvB: \/srv\/$INPUT_DIR\/make_friend_SvB.json@@SvB|" -e "s|SvB_MA: .*|SvB_MA: \/srv\/$INPUT_DIR\/make_friend_SvB.json@@SvB_MA|" analysis/metadata/HH4b_signals.yml > $OUTPUT_DIR/HH4b_signals.yml
else
    sed -e "s|SvB: .*|SvB: \/builds\/${CI_PROJECT_PATH}\/$INPUT_DIR\/make_friend_SvB.json@@SvB|" -e "s|SvB_MA: .*|SvB_MA: \/builds\/${CI_PROJECT_PATH}\/$INPUT_DIR\/make_friend_SvB.json@@SvB_MA|" analysis/metadata/HH4b_signals.yml > $OUTPUT_DIR/HH4b_signals.yml
fi
cat $OUTPUT_DIR/HH4b_signals.yml

echo "############### Running test processor"
python runner.py -t -o test_SvB_friend.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $OUTPUT_DIR/HH4b_signals.yml -m $DATASETS
ls -lR ${OUTPUT_DIR}

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi