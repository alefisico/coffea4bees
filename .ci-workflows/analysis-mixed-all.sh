#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

echo "############### Running test processor"
python runner.py  -o histMixedBkg_TT.coffea -d   TTTo2L2Nu_for_mixed TTToHadronic_for_mixed TTToSemiLeptonic_for_mixed   -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP  -op $OUTPUT_DIR -m $DATASETS

python runner.py  -o histMixedBkg_data_3b_for_mixed_kfold.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c analysis/metadata/HH4b_mixed_data.yml

sed -e "s/use_kfold: True/use_kfold: False/" "analysis/metadata/HH4b_mixed_data.yml" > /tmp/${USER}/HH4b_mixed_data_nokfold.yml
python runner.py  -o histMixedBkg_data_3b_for_mixed.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c /tmp/${USER}/HH4b_mixed_data_nokfold.yml

sed -e "s/use_kfold: True/use_kfold: False/" -e "s/use_ZZinSB: False/use_ZZinSB: True/" "analysis/metadata/HH4b_mixed_data.yml" > /tmp/${USER}/HH4b_mixed_data_ZZinSB.yml
python runner.py  -o histMixedBkg_data_3b_for_mixed_ZZinSB.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c /tmp/${USER}/HH4b_mixed_data_ZZinSB.yml

sed -e "s/use_kfold: True/use_kfold: False/" -e "s/use_ZZandZHinSB: False/use_ZZandZHinSB: True/" "analysis/metadata/HH4b_mixed_data.yml" > /tmp/${USER}/HH4b_mixed_data_ZZandZHinSB.yml
python runner.py  -o histMixedBkg_data_3b_for_mixed_ZZandZHinSB.coffea -d   data_3b_for_mixed  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS -c /tmp/${USER}/HH4b_mixed_data_ZZandZHinSB.yml

python runner.py  -o histMixedData.coffea -d    mixeddata  -p analysis/processors/processor_HH4b.py -y 2016 2017 2018    -op $OUTPUT_DIR -m $DATASETS

python runner.py  -o histSignal.coffea -d    GluGluToHHTo4B_cHHH1 ZH4b ZZ4b  -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP    -op $OUTPUT_DIR -m $DATASETS
ls

echo "############### Hist --> JSON"

python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedData.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_TT.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_kfold.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZinSB.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histMixedBkg_data_3b_for_mixed_ZZandZHinSB.coffea
python stats_analysis/convert_hist_to_json_closure.py --input $OUTPUT_DIR/histSignal.coffea

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
