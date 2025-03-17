#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"}

echo "############### Running makeweights test"
python analysis/tests/topCand_test.py

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi