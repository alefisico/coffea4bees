#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}SvB_friendtree_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#/srv/output/tmp/#\/srv\/$OUTPUT_DIR\/#" analysis/metadata/HH4b_make_friend_SvB.yml > $OUTPUT_DIR/HH4b_make_friend_SvB.yml
else
    sed -e "s#/srv/output/tmp/#\/builds\/${CI_PROJECT_PATH}\/$OUTPUT_DIR\/#" analysis/metadata/HH4b_make_friend_SvB.yml > $OUTPUT_DIR/HH4b_make_friend_SvB.yml
fi
cat $OUTPUT_DIR/HH4b_make_friend_SvB.yml

echo "############### Running test processor"
python runner.py -t -o make_friend_SvB.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $OUTPUT_DIR/HH4b_make_friend_SvB.yml -m $DATASETS
ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi