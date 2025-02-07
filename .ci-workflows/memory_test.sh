#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}memory_test"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running memory test"
sed -e "s#  workers: 4.*#  workers: 1\n  maxchunks: 1#" analysis/metadata/HH4b_signals.yml > $OUTPUT_DIR/HH4b_memory_test.yml
python base_class/tests/memory_test.py --threshold 3500 --tolerance 20 -o $OUTPUT_DIR/mprofile_ci_test --script runner.py -o test.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op local_outputs/analysis/ -m $DATASETS -c $OUTPUT_DIR/HH4b_memory_test.yml
ls $OUTPUT_DIR/mprofile_ci_test.png

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi