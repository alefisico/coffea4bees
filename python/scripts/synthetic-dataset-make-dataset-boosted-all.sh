#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} 

echo "############### Checking and creating output/skimmer directory"
OUTPUT_DIR="${DEFAULT_DIR}/synthetic_dataset_make_dataset"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/"  skimmer/metadata/declustering_boosted.yml > $OUTPUT_DIR/declustering_boosted_for_test.yml
elif [[ $(hostname) = *rogue* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/${OUTPUT_DIR}\/#"   skimmer/metadata/declustering_boosted.yml > $OUTPUT_DIR/declustering_boosted_for_test.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/${OUTPUT_DIR}\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/T3_US_FNALLPC/T3_CH_PSI/"   skimmer/metadata/declustering_boosted.yml > $OUTPUT_DIR/declustering_boosted_for_test.yml
fi
#cat $OUTPUT_DIR/declustering_boosted_for_test.yml

echo "Hostname"
echo $(hostname)

echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/make_declustered_data_boosted_4b.py -c $OUTPUT_DIR/declustering_boosted_for_test.yml -y UL18  -d data   -op $OUTPUT_DIR -o picoaod_datasets_declustered_boosted_test_UL18.yml -m metadata/datasets_HH4b_2024_v2_boosted.yml  --debug




ls -R $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
