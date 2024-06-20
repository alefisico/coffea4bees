echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
export BASE=skimmer/tests/
echo "############### Changing metadata"
sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/skimmer\/#" > ${BASE}modify_branches.yml
cat ${BASE}modify_branches.yml
echo "############### Skimming"
python runner.py -s -p ${BASE}modify_branches.py -c ${BASE}modify_branches_skimmer.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op ${BASE} -o picoAOD_modify_branches.yml -m metadata/datasets_HH4b.yml -t --debug
ls -R skimmer/
echo "############### Checking skimmer output"
python metadata/merge_yaml_datasets.py -m metadata/datasets_HH4b.yml -f ${BASE}picoAOD_modify_branches.yml -o ${BASE}picoAOD_modify_branches.yml
python runner.py -p ${BASE}modify_branches.py -c ${BASE}modify_branches_analysis.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op ${BASE} -o modify_branches.coffea -m ${BASE}picoAOD_modify_branches.yml -t --debug
