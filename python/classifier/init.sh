#!/bin/bash
parent=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
source ${parent}/autocomplete/_init.sh
chmod +x ${parent}/../run_classifier.py