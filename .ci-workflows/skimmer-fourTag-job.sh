echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata"

echo "############### Running test processor"
#time python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/HH4b_fourTag.yml -y UL18 UL17 UL16_preVFP UL16_postVFP -d data -op skimmer/metadata/ -o picoaod_datasets_fourTag_data_Run2.yml -m metadata/datasets_HH4b.yml
time python runner.py -s -p skimmer/processor/skimmer_4b.py -c skimmer/metadata/HH4b_fourTag.yml -y 2022_EE 2022_preEE 2023_BPix 2023_preBPix -d data -op skimmer/metadata/ -o picoaod_datasets_fourTag_data_Run3.yml -m metadata/datasets_HH4b_Run3.yml 
#ls -R skimmer/
cd ../
