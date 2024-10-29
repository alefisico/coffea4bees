echo "############### Including proxy"
if [ ! -f "${PWD}/proxy/x509_proxy" ]; then
    echo "Error: x509_proxy file not found!"
    exit 1
fi
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info

echo "############### Moving to python folder"
cd python/
echo "############### Checking and creating output/skimmer directory"
OUTPUT_DIR="output/synthetic_dataset_make_dataset"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/python\/output\/synthetic_dataset_make_dataset\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/"  skimmer/metadata/declustering.yml > $OUTPUT_DIR/declustering_for_test.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/output\/synthetic_dataset_make_dataset\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/declustering.yml > $OUTPUT_DIR/declustering_for_test.yml
fi
cat $OUTPUT_DIR/declustering_for_test.yml
echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c $OUTPUT_DIR/declustering_for_test.yml -y UL18  -d data GluGluToHHTo4B_cHHH1 -op $OUTPUT_DIR -o picoaod_datasets_declustered_test_UL18.yml -m metadata/datasets_HH4b.yml

# time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c $OUTPUT_DIR/declustering_for_test.yml -y UL18  -d GluGluToHHTo4B_cHHH1 -op $OUTPUT_DIR -o picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_test_UL18.yml -m metadata/datasets_HH4b.yml
ls -R $OUTPUT_DIR
cd ../