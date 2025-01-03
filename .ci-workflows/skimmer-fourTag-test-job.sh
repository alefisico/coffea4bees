#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=true ${1:-"output/"}

OUTPUT_DIR="${DEFAULT_DIR}/skimmer_fourTag_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi


echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/python\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/2024_.*/tmp\//" skimmer/metadata/HH4b_fourTag.yml > ${OUTPUT_DIR}/tmp_fourTag.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/2024_.*/tmp\//" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/HH4b_fourTag.yml > ${OUTPUT_DIR}/tmp_fourTag.yml
fi
cat ${OUTPUT_DIR}/tmp_fourTag.yml

echo "############### Running test processor"
python runner.py -s -p skimmer/processor/skimmer_4b.py -c ${OUTPUT_DIR}/tmp_fourTag.yml -y UL18 -d data -op ${OUTPUT_DIR} -o picoaod_datasets_fourTag_data_test_UL18.yml -m metadata/datasets_HH4b.yml 

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi