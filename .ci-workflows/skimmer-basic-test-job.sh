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
nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2810000/94DA5440-3B94-354B-A25B-78518A52D2D1.root"
sed -e "s#/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" metadata/datasets_HH4b.yml > metadata/datasets_ci.yml
cat metadata/datasets_ci.yml
echo "############### Skimming"
python runner.py -s -p ${BASE}modify_branches.py -c ${BASE}modify_branches_skimmer.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op ${BASE} -o picoAOD_modify_branches.yml -m metadata/datasets_ci.yml -t --debug
ls -R skimmer/
echo "############### Checking skimmer output"
python metadata/merge_yaml_datasets.py -m metadata/datasets_ci.yml -f ${BASE}picoAOD_modify_branches.yml -o ${BASE}picoAOD_modify_branches.yml
python runner.py -p ${BASE}modify_branches.py -c ${BASE}modify_branches_analysis.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op ${BASE} -o modify_branches.coffea -m ${BASE}picoAOD_modify_branches.yml -t --debug
