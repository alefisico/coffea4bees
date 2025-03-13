#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"}

INPUT_DIR="${DEFAULT_DIR}synthetic_dataset_cluster"
OUTPUT_DIR="${DEFAULT_DIR}synthetic_dataset_plot_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi
echo "############### Running test processor"
python  jet_clustering/make_jet_splitting_PDFs.py $INPUT_DIR/test_synthetic_datasets.coffea --doTest  --out $OUTPUT_DIR/jet-splitting-PDFs-test
echo "############### Checking if pdf files exist"
ls $OUTPUT_DIR/jet-splitting-PDFs-test/clustering_pdfs_vs_pT_RunII.yml 
ls $OUTPUT_DIR/jet-splitting-PDFs-test/test_sampling_pt_1b0j_1b0j_mA.pdf 

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi