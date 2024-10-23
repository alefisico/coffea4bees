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

OUTPUT_DIR="output/topreco_friendtree_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
    sed -e "s#make_top_reconstruction:.*#make_top_reconstruction: \/srv\/python\/$OUTPUT_DIR\/#" analysis/metadata/HH4b_top_reconstruction.yml > $OUTPUT_DIR/HH4b_top_reconstruction.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
    sed -e "s#make_top_reconstruction:.*#make_top_reconstruction: \/builds\/${CI_PROJECT_PATH}\/python\/$OUTPUT_DIR\/#" analysis/metadata/HH4b_top_reconstruction.yml > $OUTPUT_DIR/HH4b_top_reconstruction.yml
fi
cat $OUTPUT_DIR/HH4b_top_reconstruction.yml
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"
python runner.py -t -o top_reconstruction_friendtree.json -d data GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op $OUTPUT_DIR -c $OUTPUT_DIR/HH4b_top_reconstruction.yml -m $DATASETS
ls $OUTPUT_DIR
cd ../
