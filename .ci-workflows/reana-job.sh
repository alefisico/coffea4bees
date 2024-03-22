export REANA_SERVER_URL=https://reana.cern.ch
export REANA_ACCESS_TOKEN="${REANA_TOKEN}"
reana-client ping
echo """
##########################################################
#### THIS JOB WILL FAILED IF YOU DONT HAVE A REANA ACCOUNT
#### AND THE REANA TOKEN AS CI SECRETS VARIABLE
#### BECAUSE AT THE MOMENT REANA DOES NOT HAVE GROUP ACCOUNTS
#### BUT IT HAS TO RUN IN THE CMU CENTRAL REPO AND IT IS
#### ALLOWED TO FAILED FOR MERGE REQUEST
##########################################################
"""
set -i -e "s#condor_cores.*#condor_cores: 6#" -e "s#condor_memory.*#condor_memory: 8#" python/analysis/metadata/HH4b.yml
reana-client run -f reana.yaml -w coffea4bees
