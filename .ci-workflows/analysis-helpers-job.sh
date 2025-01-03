#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=false ${1:-"output/"}

echo "############### Running makeweights test"
python analysis/tests/topCand_test.py

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi