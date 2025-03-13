#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"}

echo "############### Running jet clustering test"
python jet_clustering/tests/test_clustering.py 

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi