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

OUTPUT_DIR="output/analysis_test_mixed_job"
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
python runner.py -t -o testMixedBkg_TT.coffea -d   TTTo2L2Nu_for_mixed TTToHadronic_for_mixed TTToSemiLeptonic_for_mixed   -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS
python runner.py -t -o testMixedBkg_data_3b_for_mixed_kfold.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2017 2018 2016  -op $OUTPUT_DIR -m $DATASETS

sed -e "s/use_kfold: True/use_kfold: False/" $DATASETS > $OUTPUT_DIR/datasets_HH4b_no_kfold.yml
python runner.py -t -o testMixedBkg_data_3b_for_mixed.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2017 2018 2016  -op $OUTPUT_DIR -m $OUTPUT_DIR/datasets_HH4b_no_kfold.yml

python runner.py -t -o testMixedData.coffea -d    mixeddata  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018 -op $OUTPUT_DIR -m $DATASETS
python runner.py -t -o testSignal_UL.coffea -d GluGluToHHTo4B_cHHH1 ZH4b ZZ4b  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS
#python runner.py -t -o testSignal_preUL.coffea -d HH4b -p analysis/processors/processor_HH4b.py -y 2016 2017 2018 -op $OUTPUT_DIR -m $DATASETS
ls $OUTPUT_DIR

echo "############### Hist --> JSON"

python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_TT.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed_kfold.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedData.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testSignal_UL.coffea
#python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testSignal_preUL.coffea

ls $OUTPUT_DIR

cd ../
