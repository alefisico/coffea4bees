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
python runner.py -t -o test.coffea -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu  ZH4b ZZ4b GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op analysis/hists/ -m $DATASETS
#python runner.py -t -o test_preUL.coffea -d HH4b -p analysis/processors/processor_HH4b.py -y 2016 2017 2018 -op analysis/hists/ -m $DATASETS
#python analysis/merge_coffea_files.py -f analysis/hists/test_UL.coffea analysis/hists/test_preUL.coffea -o analysis/hists/test.coffea
ls
cd ../
