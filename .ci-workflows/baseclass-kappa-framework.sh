#!/bin/bash
source .ci-workflows/set_initial_variables.sh do_proxy=false ${1:-"output/"}

echo "############### Running kappa framework test"
python -m base_class.tests.kappa_framework
