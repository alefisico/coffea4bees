#!/bin/bash

echo "############### Setting up environment"
if [ "$1" == "do_proxy=true" ]; then
    echo "############### Including proxy"
    if [ ! -f "${PWD}/proxy/x509_proxy" ]; then
        echo "Error: x509_proxy file not found!"
        exit 1
    fi
    export X509_USER_PROXY=${PWD}/proxy/x509_proxy
fi

echo "############### Checking proxy"
voms-proxy-info

return_to_base=false
if [ "$(basename "$PWD")" == "python" ]; then
    echo "You are in the python directory."
else
    return_to_base=true
    echo "############### Moving to python folder"
    cd python/
fi

echo "############### Checking and creating base output directory"
DEFAULT_DIR=${2}
echo "The base output directory is: $DEFAULT_DIR"

echo "############### Checking datasets"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
else
    DATASETS=metadata/datasets_HH4b_cernbox.yml
fi
echo "The datasets file is: $DATASETS"
