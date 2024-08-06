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

#time python runner.py  -o test_declustering_nominal.coffea     -d data  -p analysis/processors/processor_decluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS 
python runner.py -t   -o test_declustering_declustered.coffea -d data  -p analysis/processors/processor_decluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/decluster.yml

# time python runner.py    -o declustering_declustered.coffea -d data  -p analysis/processors/processor_decluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/decluster.yml
# time python runner.py    -o declustering_declustered_signal.coffea -d GluGluToHHTo4B_cHHH1  -p analysis/processors/processor_decluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS -c analysis/metadata/decluster.yml
# time python runner.py    -o declustering_nominal_signal.coffea -d GluGluToHHTo4B_cHHH1  -p analysis/processors/processor_decluster_4b.py -y UL18  -op analysis/hists/ -m $DATASETS 
# python  jet_clustering/compare_datasets.py analysis/hists/test_declustering_declustered.coffea  analysis/hists/test_declustering_nominal.coffea  --out analysis/plots_test_synthetic_datasets
ls
#cd ../
