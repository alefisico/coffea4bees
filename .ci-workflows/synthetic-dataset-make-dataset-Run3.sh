#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

echo "############### Checking and creating output/skimmer directory"
OUTPUT_DIR="${DEFAULT_DIR}/synthetic_dataset_make_dataset_Run3"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Changing metadata"
echo "############### Overwritting datasets"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/python\/${OUTPUT_DIR}\/synthetic_dataset_make_dataset_Run3\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/"  skimmer/metadata/declustering_noTT_subtraction.yml > $OUTPUT_DIR/declustering_for_test.yml
    DATASETS=metadata/datasets_HH4b_Run3_fourTag_v3.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/${OUTPUT_DIR}\/synthetic_dataset_make_dataset_Run3\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/declustering_noTT_subtraction.yml > $OUTPUT_DIR/declustering_for_test.yml
    DATASETS=metadata/datasets_HH4b_Run3_cernbox.yml
fi
cat $OUTPUT_DIR/declustering_for_test.yml

echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c $OUTPUT_DIR/declustering_for_test.yml -y 2023_BPix  -d data  -op $OUTPUT_DIR -o picoaod_datasets_declustered_test_2023_BPix.yml -m $DATASETS

# time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c $OUTPUT_DIR/declustering_for_test.yml -y UL18  -d GluGluToHHTo4B_cHHH1 -op $OUTPUT_DIR -o picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_test_UL18.yml -m metadata/datasets_HH4b.yml
ls -R $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
