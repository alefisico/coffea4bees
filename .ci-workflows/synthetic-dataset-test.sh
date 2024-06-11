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
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o test_synthetic_datasets.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/cluster_4b.yml
#python runner.py -t -o test_synthetic_datasets.coffea -d data  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -m $DATASETS
ls
cd ../
