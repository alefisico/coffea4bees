#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=true ${1:-"output/"}

OUTPUT_DIR="${DEFAULT_DIR}/analysis_testAll_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi


echo "############### Running test processor"
python runner.py -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu ZZ4b ZH4b GluGluToHHTo4B_cHHH1   -p analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP -o histAll.coffea -op ${OUTPUT_DIR}
ls
python analysis/tests/cutflow_test.py   --inputFile ${OUTPUT_DIR}/histAll.coffea --knownCounts analysis/tests/histAllCounts.yml

# python runner.py -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu ZZ4b ZH4b GluGluToHHTo4B_cHHH1 -c analysis/metadata/HH4b_noFvT.yml   -p analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP -o histAll_noFvT.coffea -op hists/

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi