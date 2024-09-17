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
    echo "############### Modifying config"
    sed -e "s|# friend_.*|friend_trigWeight: \/builds\/${CI_PROJECT_PATH}\/python\/analysis\/trigger_weights\/|" analysis/metadata/HH4b.yml > analysis/metadata/trigger_weights_HH4b.yml
    cat analysis/metadata/trigger_weights_HH4b.yml
fi
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o test_trigWeight.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op analysis/trigger_weights/ -c analysis/metadata/trigger_weights_HH4b.yml -m $DATASETS
ls
cd ../
