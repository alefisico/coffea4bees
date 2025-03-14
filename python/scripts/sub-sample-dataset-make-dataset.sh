#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/sub_sample_dataset_make_dataset"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/python\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 5/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 100000/"  skimmer/metadata/sub_sampling_MC.yml > ${OUTPUT_DIR}/sub_sampling_MC_for_test.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/sub_sampling_MC.yml > ${OUTPUT_DIR}/sub_sampling_MC_for_test.yml
fi
cat ${OUTPUT_DIR}/sub_sampling_MC_for_test.yml

echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/sub_sample_MC.py -c ${OUTPUT_DIR}/sub_sampling_MC_for_test.yml -y UL18  -d TTToHadronic -op ${OUTPUT_DIR} -o picoaod_datasets_TTToHadronic_pseudodata_test_UL18.yml -m metadata/datasets_HH4b.yml
ls -R ${OUTPUT_DIR}

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi