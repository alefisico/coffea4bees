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


python runner.py -t -o test_synthetic_datasets_all.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/cluster_4b.yml
# time python runner.py  -o synthetic_datasets_all.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/cluster_4b.yml
# python runner.py -t -o test_synthetic_datasets_upto6j.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/cluster_and_decluster.yml
# time python runner.py  -o synthetic_datasets_upto6j.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/cluster_and_decluster.yml

# python  jet_clustering/make_jet_splitting_PDFs.py analysis/hists/test_synthetic_datasets_4j_and_5j.coffea  --out jet_clustering/jet-splitting-PDFs-00-02-00 

ls
#cd ../
