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

INPUT_DIR="output/synthetic_dataset_make_dataset"
OUTPUT_DIR="output/synthetic_dataset_analyze"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi


# echo "############### Modifying dataset file with skimmer ci output"
# cat metadata/datasets_ci.yml
# python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f skimmer/metadata/picoaod_datasets_declustered_data_test_UL18A.yml  -o metadata/datasets_synthetic_seed17_test.yml

#/builds/johnda/coffea4bees/python/skimmer/GluGluToHHTo4B_cHHH1_UL18/picoAOD_seed5.root
#/builds/johnda/coffea4bees/python
#johnda/coffea4bees
#/builds/python/skimmer/GluGluToHHTo4B_cHHH1_UL18/picoAOD_seed5.root


# echo "############### Changing metadata"
# sed -e "s/apply_FvT.*/apply_FvT: false/" -e "s/apply_trig.*/apply_trigWeight: false/" -e "s/run_SvB.*/run_SvB: false/"  analysis/metadata/HH4b.yml > $OUTPUT_DIR/tmp.yml
# cat $OUTPUT_DIR/tmp.yml
#echo "############### Running test processor"
#time python runner.py -o test_synthetic_data_test.coffea -d data -p analysis/processors/processor_HH4b.py -y UL18  -op $OUTPUT_DIR/ -c $OUTPUT_DIR/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_seed17.yml


echo "############### Running test processor "
# python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f skimmer/metadata/picoaod_datasets_declustered_test_UL18.yml  -o metadata/datasets_synthetic_test.yml
# python metadata/merge_yaml_datasets.py -m metadata/datasets_synthetic_seed17.yml -f skimmer/metadata/picoaod_datasets_declustered_GluGluToHHTo4B_cHHH1_Run2_seed17.yml -o metadata/datasets_synthetic_seed17.yml
#cat metadata/datasets_synthetic_test.yml

time python runner.py -o test_synthetic_datasets.coffea -d synthetic_data synthetic_mc_GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18  -op $OUTPUT_DIR/ -c analysis/metadata/HH4b_synthetic_data.yml -m metadata/datasets_synthetic_test.yml

# time python runner.py -o test_synthetic_datasets.coffea -d data GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18  -op $OUTPUT_DIR/ -c analysis/metadata/HH4b_synthetic_data.yml -m $OUTPUT_DIR/datasets_synthetic_test.yml
cd ../
