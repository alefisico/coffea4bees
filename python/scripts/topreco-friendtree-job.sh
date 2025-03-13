#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}topreco_friendtree_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#make_top_reconstruction:.*#make_top_reconstruction: \/srv\/python\/$OUTPUT_DIR\/#" analysis/metadata/HH4b_top_reconstruction.yml > $OUTPUT_DIR/HH4b_top_reconstruction.yml
else
    sed -e "s#make_top_reconstruction:.*#make_top_reconstruction: \/builds\/${CI_PROJECT_PATH}\/python\/$OUTPUT_DIR\/#" analysis/metadata/HH4b_top_reconstruction.yml > $OUTPUT_DIR/HH4b_top_reconstruction.yml
fi
cat $OUTPUT_DIR/HH4b_top_reconstruction.yml

echo "############### Running test processor"
python runner.py -t -o top_reconstruction_friendtree.json -d data GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $OUTPUT_DIR/HH4b_top_reconstruction.yml -m $DATASETS
ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi