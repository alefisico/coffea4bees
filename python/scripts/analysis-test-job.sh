#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/analysis_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Modifying config"
sed -e "s|hist_cuts: .*|hist_cuts: [ passPreSel, passSvB, failSvB ]|" analysis/metadata/HH4b.yml > $OUTPUT_DIR/HH4b.yml
cat $OUTPUT_DIR/HH4b.yml

echo "############### Running test processor"
python runner.py -t -o test_databkgs.coffea -d data TTToHadronic TTToSemiLeptonic TTTo2L2Nu ggZH4b ZH4b ZZ4b -p analysis/processors/processor_HH4b.py -y UL17 UL18 UL16_preVFP UL16_postVFP -op $OUTPUT_DIR -m $DATASETS -c $OUTPUT_DIR/HH4b.yml

ls $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi
