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

OUTPUT_DIR="output/analysis_systematics_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying corrections for ci"
sed -e "/Absolute_/d" -e "/BBEC1/d" -e "/EC2/d" -e "/- HF/d" -e "/- Relative/d" -e "/- hf/d" -e "/- lf/d" analysis/metadata/corrections.yml > $OUTPUT_DIR/corrections_ci.yml
cat $OUTPUT_DIR/corrections_ci.yml

echo "############### Modifying config"
sed -e "s#corrections_metadata:.*#corrections_metadata: $OUTPUT_DIR/corrections_ci.yml#" analysis/metadata/HH4b_systematics.yml > $OUTPUT_DIR/HH4b_systematics_ci.yml
cat $OUTPUT_DIR/HH4b_systematics_ci.yml

echo "############### Modifying datasets"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
fi
echo "############### Running datasets from " $DATASETS

echo "############### Running test processor"
python runner.py -t -o test_systematics.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR/ -m $DATASETS -c $OUTPUT_DIR/HH4b_systematics_ci.yml
#python runner.py -t -o test_systematics_preUL.coffea -d HH4b -p analysis/processors/processor_HH4b.py -y 2018 -op $OUTPUT_DIR/ -m $DATASETS -c $OUTPUT_DIR/HH4b_systematics_ci.yml
#python analysis/merge_coffea_files.py -f $OUTPUT_DIR/test_systematics_UL.coffea $OUTPUT_DIR/test_systematics_preUL.coffea -o $OUTPUT_DIR/test_systematics.coffea
ls $OUTPUT_DIR
cd ../
