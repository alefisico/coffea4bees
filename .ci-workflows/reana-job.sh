if [[ $(hostname) = *lxplus* ]]; then
    echo """Running local in lxplus"""
    export workflow_name="workflow"
    export CMD='reana-client'
else
    export REANA_SERVER_URL=https://reana.cern.ch
    export REANA_ACCESS_TOKEN="${REANA_TOKEN}"
    export workflow_name="coffea4bees"
    #reana-client ping
    echo """
    ##########################################################
    #### THIS JOB WILL FAILED IF YOU DONT HAVE A REANA ACCOUNT
    #### AND THE REANA TOKEN AS CI SECRETS VARIABLE
    #### BECAUSE AT THE MOMENT REANA DOES NOT HAVE GROUP ACCOUNTS
    #### BUT IT HAS TO RUN IN THE CMU CENTRAL REPO AND IT IS
    #### ALLOWED TO FAILED FOR MERGE REQUEST
    ##########################################################
    """
    docker pull docker.io/reanahub/reana-client:0.9.3
    export CMD="docker run -it --env REANA_SERVER_URL --env REANA_ACCESS_TOKEN docker.io/reanahub/reana-client:0.9.3"
fi
sed -e "#hash:.*#hash: "$(git rev-parse HEAD)"#" -i .reana_workflows/inputs.yaml
git diff HEAD > gitdiff.txt
cat .reana_workflows/inputs.yaml
$CMD run -f reana.yaml -w ${workflow_name}
