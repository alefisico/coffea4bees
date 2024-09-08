echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/python\/skimmer\/test\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/"  skimmer/metadata/declustering.yml > skimmer/metadata/declustering_for_test.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/skimmer\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/declustering.yml > skimmer/metadata/declustering_for_test.yml
fi
cat skimmer/metadata/declustering_for_test.yml
echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustering_for_test.yml -y UL18  -d data GluGluToHHTo4B_cHHH1 -op skimmer/metadata/ -o picoaod_datasets_declustered_test_UL18.yml -m metadata/datasets_HH4b.yml

# time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustering_for_test.yml -y UL18  -d GluGluToHHTo4B_cHHH1 -op skimmer/metadata/ -o picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_test_UL18.yml -m metadata/datasets_HH4b.yml
ls -R skimmer/
cd ../
