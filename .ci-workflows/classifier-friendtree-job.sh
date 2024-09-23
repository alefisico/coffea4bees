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
    sed -e "s#make_.*#make_classifier_input: \/builds\/${CI_PROJECT_PATH}\/python\/analysis\/classifier_inputs\/#" -i analysis/metadata/HH4b_classifier_inputs.yml
    cat analysis/metadata/HH4b_classifier_inputs.yml
fi
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o dummy.coffea -d data GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op analysis/classifier_inputs/ -c analysis/metadata/HH4b_classifier_inputs.yml -m $DATASETS
ls
cd ../
