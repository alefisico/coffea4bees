echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Running test processor"
python runner.py -t -o test.coffea -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu  HH4b ZH4b ZZ4b -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/
ls
cd ../
