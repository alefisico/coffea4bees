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

OUTPUT_DIR="output/synthetic_dataset_cluster"
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
echo "############### Running datasets from " $DATASETS
echo "############### Running test processor"


python runner.py -t -o test_synthetic_datasets.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/cluster_4b.yml
# python runner.py -t -o test_cluster_synthetic_data.coffea -d synthetic_data  -p analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/cluster_4b.yml

# time python runner.py  -o cluster_data_Run2.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/cluster_4b_noTTSubtraction.yml
# time python runner.py  -o cluster_data_Run2_noTT.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/cluster_4b.yml
# time python runner.py  -o cluster_synthetic_data_Run2.coffea -d synthetic_data  -p analysis/processors/processor_cluster_4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/cluster_4b_noTTSubtraction.yml
# python runner.py -t -o test_synthetic_datasets_upto6j.coffea -d data  -p analysis/processors/processor_cluster_4b.py -y UL18  -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/cluster_and_decluster.yml


# python  jet_clustering/make_jet_splitting_PDFs.py $OUTPUT_DIR/test_synthetic_datasets_4j_and_5j.coffea  --out jet_clustering/jet-splitting-PDFs-00-02-00 

ls
cd ../
