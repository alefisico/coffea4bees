#!/usr/bin/env bash

echo """
************************************************************************************************
************************************************************************************************
This script will be deprecated soon.
Please use the run_coffea function in coffea4bees/python/run_container instead.
To learn about the script do to python and run: ./run_container --help
To run run_container as this script does, do: ./run_container combine <command>
************************************************************************************************
************************************************************************************************
"""

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <container> [command...]"
  exit 1
fi

CONTAINER=$1
shift

export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security/vomses:/etc/vomses,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/tmp
export APPTAINER_CACHEDIR="/tmp/$(whoami)/apptainer_cache"
export APPTAINER_TMPDIR="/tmp/.apptainer/"
export MPLCONFIGDIR="/tmp/$(whoami)/.config/matplotlib"

APPTAINER_SHELL=$(which bash) apptainer exec -B ${PWD}:/home/cmsusr/coffea4bees \
    --pwd /home/cmsusr/coffea4bees/  \
    $CONTAINER \
    /bin/bash -c "source /cvmfs/cms.cern.ch/cmsset_default.sh && \
    cd /home/cmsusr/CMSSW_11_3_4/ && cmsenv && cd /home/cmsusr/coffea4bees/ && \
    $*"