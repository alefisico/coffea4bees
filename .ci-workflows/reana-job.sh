if [[ $(hostname) = *lxplus* ]]; then
    echo """Running local in lxplus"""
    export workflow_name="workflow"
else
    export REANA_SERVER_URL=https://reana.cern.ch
    export REANA_ACCESS_TOKEN="${REANA_TOKEN}"
    export workflow_name="coffea4bees"
    virtualenv ~/.virtualenvs/reana
    source ~/.virtualenvs/reana/bin/activate
    pip install reana-client
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
fi
sed -e "#hash:.*#hash: "$(git rev-parse HEAD)"#" -i .reana_workflows/inputs.yaml
git diff HEAD > gitdiff.txt
cat .reana_workflows/inputs.yaml
reana-client run -f reana.yaml -w ${workflow_name}
