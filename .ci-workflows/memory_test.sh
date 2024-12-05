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

OUTPUT_DIR="output/memory_test"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

# echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
fi

echo "############### Running memory test"
python base_class/tests/memory_test.py --threshold 3689.422 -o $OUTPUT_DIR/mprofile_ci_test --script runner.py -o test.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op local_outputs/analysis/ -m $DATASETS
ls $OUTPUT_DIR/mprofile_ci_test.png
cd ../

