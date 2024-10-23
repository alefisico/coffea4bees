echo "############### Including proxy"
if [ ! -f "${PWD}/proxy/x509_proxy" ]; then
    echo "Error: x509_proxy file not found!"
    exit 1
fi

echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy

echo "############### Checking proxy"
voms-proxy-info

echo "############### Moving to python folder"
cd python/

OUTPUT_DIR="output/weights_trigger_friendtree_job"
echo "############### Checking and creating output/skimmer directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
    sed -e "s#make_.*#make_classifier_input: \/srv\/python\/$OUTPUT_DIR\/#" analysis/metadata/trigger_weights.yml > $OUTPUT_DIR/trigger_weights.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
    sed -e "s#make_.*#make_classifier_input: \/builds\/${CI_PROJECT_PATH}\/python\/$OUTPUT_DIR\/#" analysis/metadata/trigger_weights.yml > $OUTPUT_DIR/trigger_weights.yml
fi
cat $OUTPUT_DIR/trigger_weights.yml

echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o trigger_weights_friends.json -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_trigger_weights.py -y UL18 -op $OUTPUT_DIR  -c $OUTPUT_DIR/trigger_weights.yml -m $DATASETS
ls $OUTPUT_DIR
cd ../
