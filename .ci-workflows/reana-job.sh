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
sed -e 's/hash:.*/hash: '"$(git rev-parse HEAD)"'/' -i .reana_workflows/inputs.yaml
sed -e 's/diff:.*/diff: '"$(git diff HEAD)"'/' -i .reana_workflows/inputs.yaml
cat .reana_workflows/inputs.yaml
reana-client run -f reana.yaml -w coffea4bees
