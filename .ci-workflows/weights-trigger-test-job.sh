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
    sed -e "s#make_.*#make_classifier_input: \/builds\/${CI_PROJECT_PATH}\/python\/analysis\/trigger_weights\/#" > analysys/metadata/trigger_weights.yml
    cat analysis/metadata/trigger_weights.yml
fi
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o dummy.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_trigger_weights.py -y UL18 -op analysis/trigger_weights/ -c analysis/metadata/trigger_weights.yml -m $DATASETS
ls
cd ../
