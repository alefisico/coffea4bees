#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"}

echo "############### Running kappa framework test"
python -m base_class.tests.kappa_framework
