echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Modifying config"
sed -e "s/condor_cores.*/condor_cores: 6/" -e "s/condor_memory.*/condor_memory: 8GB/" -i analysis/metadata/unsup.yml
cat analysis/metadata/unsup4b.yml
echo "############### Running test processor"
python runner.py -t -o test_unsup.coffea -d mixeddata data_3b_for_mixed TTToHadronic TTToSemiLeptonic TTTo2L2Nu -p analysis/processors/processor_unsup.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -m analysis/metadata/unsup4b.yml -c analysis/metadata/unsup4b.yml
ls
cd ../
