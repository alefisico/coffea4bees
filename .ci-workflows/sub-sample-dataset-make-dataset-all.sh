echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata"
echo "############### Running test processor"
time python runner.py -s -p skimmer/processor/sub_sample_MC.py -c skimmer/metadata/sub_sampling_MC.yml -y UL17 UL18 UL16_preVFP UL16_postVFP  -d TTToHadronic TTToSemiLeptonic TTTo2L2Nu -op skimmer/metadata/ -o picoaod_datasets_TT_pseudodata_Run2.yml -m metadata/datasets_HH4b.yml
ls -R skimmer/
cd ../
