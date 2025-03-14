#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}analysis_test_mixed_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
sed -e "s|run_dilep_ttbar_crosscheck: true|run_dilep_ttbar_crosscheck: false|" analysis/metadata/HH4b.yml > $OUTPUT_DIR/HH4b.yml
cat $OUTPUT_DIR/HH4b.yml

echo "############### Running test processor"
python runner.py -t -o testMixedBkg_TT.coffea -d   TTTo2L2Nu_for_mixed TTToHadronic_for_mixed TTToSemiLeptonic_for_mixed   -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS -c $OUTPUT_DIR/HH4b.yml
python runner.py -t -o testMixedBkg_data_3b_for_mixed_kfold.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2017 2018 2016  -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/HH4b_mixed_data.yml

python runner.py -t -o testMixedBkg_data_3b_for_mixed.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2017 2018 2016  -op $OUTPUT_DIR -m $DATASETS -c $OUTPUT_DIR/HH4b.yml

python runner.py -t -o testMixedData.coffea -d    mixeddata  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018 -op $OUTPUT_DIR -m $DATASETS -c $OUTPUT_DIR/HH4b.yml
python runner.py -t -o testSignals.coffea -d ZH4b ZZ4b  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS -c $OUTPUT_DIR/HH4b.yml
python runner.py -t -o testSignals_HH4b.coffea -d GluGluToHHTo4B_cHHH1  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/HH4b_signals.yml
python analysis/tools/merge_coffea_files.py -f $OUTPUT_DIR/testSignals_HH4b.coffea $OUTPUT_DIR/testSignals.coffea -o $OUTPUT_DIR/testSignal_UL.coffea
ls $OUTPUT_DIR

echo "############### Hist --> JSON"

python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_TT.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed_kfold.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedBkg_data_3b_for_mixed.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testMixedData.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testSignal_UL.coffea
#python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/testSignal_preUL.coffea

ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi