echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustering.yml -y UL17 UL18 UL16_preVFP UL16_postVFP -d data -op skimmer/metadata/ -o picoaod_datasets_declustered_data_Run2_seed17.yml -m metadata/datasets_HH4b.yml   # --dask
# time python runner.py -s -p skimmer/processor/make_declustered_data_4b.py -c skimmer/metadata/declustering_signal.yml -y UL17 UL18 UL16_preVFP UL16_postVFP -d GluGluToHHTo4B_cHHH1 -op skimmer/metadata/ -o picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_Run2_seed17.yml -m metadata/datasets_HH4b.yml

ls -R skimmer/
cd ../
