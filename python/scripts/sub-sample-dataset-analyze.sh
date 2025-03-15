#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/sub_sample_dataset_analyze"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ $(hostname) = *fnal* ]]; then
    echo "No changing files"
else
    echo "############### Modifying previous dataset file (to read local files)"
    ls -lR skimmer/
    cat skimmer/metadata/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml
    echo "TEST"
    pwd
    echo ${CI_PROJECT_PATH}
    sed "s|\/builds/$CI_PROJECT_PATH\/python\/||g"  skimmer/metadata/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml > ${OUTPUT_DIR}/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml
    echo "NEW"
    cat ${OUTPUT_DIR}/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml
fi


# echo "############### Changing metadata"
# sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/"  analysis/metadata/HH4b.yml > ${OUTPUT_DIR}/tmp.yml
# cat analysis/metadata/tmp.yml
#echo "############### Running test processor"
#time python runner.py -o test_synthetic_data_test.coffea -d data -p analysis/processors/processor_HH4b.py -y UL18  -op ${OUTPUT_DIR} -c analysis/metadata/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_seed17.yml


echo "############### Running test processor "
# python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f ${OUTPUT_DIR}/picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml  -o ${OUTPUT_DIR}/datasets_TT_pseudodata_test.yml
# python metadata/merge_yaml_datasets.py -m metadata/datasets_synthetic_seed17.yml -f ${OUTPUT_DIR}/picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_Run2_seed17.yml -o ${OUTPUT_DIR}/datasets_synthetic_seed17.yml
#cat ${OUTPUT_DIR}/datasets_synthetic_test.yml
time python runner.py -o test_TT_pseudodata_datasets.coffea -d ps_data_TTToHadronic -p analysis/processors/processor_HH4b.py -y UL18  -op ${OUTPUT_DIR} -c analysis/metadata/HH4b_ps_data.yml -m metadata/datasets_TT_pseudodata_test.yml

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi