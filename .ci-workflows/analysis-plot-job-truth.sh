#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=false ${1:-"output/"}

OUTPUT_DIR="${DEFAULT_DIR}/analysis_plot_job_truth"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running test processor"
python  plots/makePlotsTruthStudy.py analysis/hists/testTruth.coffea -m plots/metadata/plotsSignal.yml --out ${OUTPUT_DIR}
echo "############### Checking if pdf files exist"
ls ${OUTPUT_DIR}/RunII/pass4GenBJets00/fourTag/SR/otherGenJet00_pt.pdf

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi