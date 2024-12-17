echo "############### Including proxy"
if [ ! -f "${PWD}/proxy/x509_proxy" ]; then
    echo "Error: x509_proxy file not found!"
    exit 1
fi
export X509_USER_PROXY=${PWD}/proxy/x509_proxy

echo "############### Checking proxy"
voms-proxy-info

echo "############### Moving to python folder"
cd python/

INPUT_DIR="output/weights_trigger_friendtree_job"
OUTPUT_DIR="output/weights_trigger_analysis_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
    sed -e "s|trigWeight: .*|trigWeight: \/srv\/python\/$INPUT_DIR\/trigger_weights_friends.json@@trigWeight|" analysis/metadata/HH4b.yml > $OUTPUT_DIR/trigger_weights_HH4b.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
    sed -e "s|trigWeight: .*|trigWeight: \/builds\/${CI_PROJECT_PATH}\/python\/$INPUT_DIR\/trigger_weights_friends.json@@trigWeight|" analysis/metadata/HH4b.yml > $OUTPUT_DIR/trigger_weights_HH4b.yml
fi
cat $OUTPUT_DIR/trigger_weights_HH4b.yml
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o test_trigWeight.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $OUTPUT_DIR/trigger_weights_HH4b.yml -m $DATASETS
ls
cd ../
