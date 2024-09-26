echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/python\/skimmer\/test\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/2024_.*/tmp\//" skimmer/metadata/HH4b_fourTag.yml > skimmer/metadata/tmp_fourTag.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/skimmer\/#" -e "s/\#max.*/maxchunks: 1/" -e "s/\#test.*/test_files: 1/" -e "s/\workers:.*/workers: 1/" -e "s/chunksize:.*/chunksize: 1000/" -e "s/2024_.*/tmp\//" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/HH4b_fourTag.yml > skimmer/metadata/tmp_fourTag.yml
fi
cat skimmer/metadata/tmp_fourTag.yml
echo "############### Running test processor"
python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/tmp_fourTag.yml -y UL18 -d data -op skimmer/metadata/ -o picoaod_datasets_fourTag_data_test_UL18.yml -m metadata/datasets_HH4b.yml 
#ls -R skimmer/
cd ../
