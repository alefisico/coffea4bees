#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}memory_test"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running memory test"
python base_class/tests/memory_test.py --threshold 3513.457 -o $OUTPUT_DIR/mprofile_ci_test --script runner.py -o test.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op local_outputs/analysis/ -m $DATASETS -c analysis/metadata/HH4b_signals.yml
ls $OUTPUT_DIR/mprofile_ci_test.png

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi