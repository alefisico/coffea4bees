echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata"

echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/HH4b_fourTag.yml -y UL18 UL17 UL16_preVFP UL16_postVFP -d data -op skimmer/metadata/ -o picoaod_datasets_fourTag_data_Run2.yml -m metadata/datasets_HH4b.yml 
#ls -R skimmer/
cd ../
