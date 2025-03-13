#!/bin/bash
source .ci-workflows/set_initial_variables.sh --output ${1:-"output/"}

echo "############### Running trigger emulator test"
python -m unittest base_class.tests.test_trigger_emulator
cd ../

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi