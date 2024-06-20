#!/bin/bash

# Function to handle argument parsing
parse_arguments() {

  # Set defaults for flags
  coffea4bees=false
  combine=false

  # Process arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --coffea4bees)
        coffea4bees=true
        shift
        ;;
      --combine)
        combine=true
        shift
        ;;
      *)
        echo "Invalid argument: '$1'"
        ;;
    esac
  done
}

# Parse arguments
parse_arguments "$@"

export APPTAINER_CACHEDIR="/tmp/$(whoami)/singularity"
export APPTAINER_TMPDIR="/tmp/.apptainer/"

if [ "$coffea4bees" = true ]; then

    echo "In lxplus you need to change the default version of python. Run before anything:
alias python=python3.10"

    APPTAINER_SHELL=$(which bash) apptainer shell -B /afs -B /eos -B /cvmfs -B ${PWD}:/srv --pwd /srv  /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/coffea4bees:latest 

elif [ "$combine" = true ]; then

    cat <<EOF > python/stats_analysis/set_cmssw.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /home/cmsusr/CMSSW_11_3_4/
cmsenv  # Ignore errors
cd /home/cmsusr/coffea4bees/python/stats_analysis/
EOF

    echo "Do not forget to run:
source python/stats_analysis/set_cmssw.sh"

    APPTAINER_SHELL=$(which bash) apptainer shell -B /afs -B /eos -B /cvmfs -B ${PWD}:/home/cmsusr/coffea4bees --pwd /home/cmsusr/coffea4bees/ /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_11_3_4-combine_v9.1.0-harvester_v2.1.0 

fi
