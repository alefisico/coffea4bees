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

INPUT_DIR="output/skimmer_test_job"
OUTPUT_DIR="output/skimmer_analysis_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

if [[ $(hostname) = *fnal* ]]; then
    echo "No changing files"
else
    echo "############### Modifying previous dataset file (to read local files)"
    sed -i "s|\/builds/$CI_PROJECT_PATH\/python\/||g" $INPUT_DIR/picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml
    cat $INPUT_DIR/picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml
fi
echo "############### Modifying dataset file with skimmer ci output"
python metadata/merge_yaml_datasets.py -m $INPUT_DIR/datasets_HH4b.yml -f $INPUT_DIR/picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml -o $OUTPUT_DIR/datasets_HH4b.yml
cat $OUTPUT_DIR/datasets_HH4b.yml

echo "############### Changing metadata"
sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/" -e "s/top_reco.*/top_reconstruction_override: 'fast'/"  analysis/metadata/HH4b.yml > $OUTPUT_DIR/HH4b.yml
cat $OUTPUT_DIR/HH4b.yml

echo "############### Running test processor"
python runner.py -o test_skimmer.coffea -d GluGluToHHTo4B_cHHH0 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $OUTPUT_DIR/HH4b.yml -m $OUTPUT_DIR/datasets_HH4b.yml
cd ../
