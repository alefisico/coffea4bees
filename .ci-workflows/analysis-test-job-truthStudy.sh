echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
    #echo "############### Modifying config"
    #sed -e "s/condor_cores.*/condor_cores: 6/" -e "s/condor_memory.*/condor_memory: 8GB/" -i analysis/metadata/HH4b.yml
    #cat analysis/metadata/HH4b.yml
fi
DATASETS=metadata/datasets_HH4b_cernbox.yml
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t    -o testTruth.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_genmatch_HH4b.py -y UL18  -op analysis/hists/ -m $DATASETS  -c analysis/metadata/HH4b_genmatch.yml
# python runner.py     -o testTruth.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_genmatch_HH4b.py -y UL18  -op analysis/hists/ -m $DATASETS  -c analysis/metadata/HH4b_genmatch.yml 

cd ../
