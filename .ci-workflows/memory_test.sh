#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=true ${1:-"output/"}

OUTPUT_DIR="${DEFAULT_DIR}memory_test"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Running memory test"
python base_class/tests/memory_test.py --threshold 3689.422 -o $OUTPUT_DIR/mprofile_ci_test --script runner.py -o test.coffea -d GluGluToHHTo4B_cHHH1 -p analysis/processors/processor_HH4b.py -y UL18 -op local_outputs/analysis/ -m $DATASETS
ls $OUTPUT_DIR/mprofile_ci_test.png

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi