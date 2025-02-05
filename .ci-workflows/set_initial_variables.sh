#!/bin/bash

# Default values
do_proxy=false
output="output/"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --do_proxy) do_proxy=true ;;
        --output) output="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "############### Setting up environment"
if [ "$do_proxy" = true ]; then
    echo "############### Including proxy"
    if [ ! -f "${PWD}/proxy/x509_proxy" ]; then
        echo "Error: x509_proxy file not found!"
        exit 1
    fi
    export X509_USER_PROXY=${PWD}/proxy/x509_proxy
    echo "############### Checking proxy"
    voms-proxy-info
fi

return_to_base=false
if [ "$(basename "$PWD")" == "python" ]; then
    echo "You are in the python directory."
else
    return_to_base=true
    echo "############### Moving to python folder"
    cd python/
fi

echo "############### Checking and creating base output directory"
DEFAULT_DIR=${output}
echo "The base output directory is: $DEFAULT_DIR"

echo "############### Checking datasets"
if [[ $(hostname) = *fnal* ]]; then
    DATASETS=metadata/datasets_HH4b.yml
else
    # DATASETS=metadata/datasets_HH4b_cernbox.yml
    DATASETS=metadata/datasets_HH4b.yml
fi
echo "The datasets file is: $DATASETS"
