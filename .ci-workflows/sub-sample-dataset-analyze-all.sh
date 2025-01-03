#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=true ${1:-"output/"}

OUTPUT_DIR="${DEFAULT_DIR}/sub_sample_dataset_analyze_all"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

# echo "############### Changing metadata"
# sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/"  analysis/metadata/HH4b.yml > ${OUTPUT_DIR}/tmp.yml
# cat ${OUTPUT_DIR}/tmp.yml
#echo "############### Running test processor"
#time python runner.py -o test_synthetic_data_test.coffea -d data -p analysis/processors/processor_HH4b.py -y UL18  -op ${OUTPUT_DIR} -c analysis/metadata/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_seed17.yml


echo "############### Running test processor "
# python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f ${OUTPUT_DIR}/picoaod_datasets_TT_pseudodata_Run2.yml  -o ${OUTPUT_DIR}datasets_TT_pseudodata_Run2.yml
#cat ${OUTPUT_DIR}/datasets_synthetic_test.yml
time python runner.py -o TT_pseudodata_datasets.coffea -d ps_data_TTToSemiLeptonic ps_data_TTTo2L2Nu ps_data_TTToHadronic -p analysis/processors/processor_HH4b.py -y UL18 UL17 UL16_preVFP UL16_postVFP  -op ${OUTPUT_DIR} -c analysis/metadata/HH4b_rerun_SvB.yml -m metadata/datasets_TT_pseudodata_Run2.yml
#time python runner.py -o histAll_TT.coffea            -d TTToHadronic TTToSemiLeptonic TTTo2L2Nu                         -p analysis/processors/processor_HH4b.py  -y UL17 UL18 UL16_preVFP UL16_postVFP  -op analysis/hists/ -c analysis/metadata/HH4b_rerun_SvB.yml 

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi