echo "############### Including proxy"
export X509_USER_PROXY=${PWD}/proxy/x509_proxy
echo "############### Checking proxy"
voms-proxy-info
echo "############### Moving to python folder"
cd python/
echo "############### Changing metadata"
sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/skimmer\/#" > skimmer/tests/modify_branches.yml
cat skimmer/tests/modify_branches.yml
echo "############### Running skimmer test processor"
python runner.py -s -p skimmer/tests/modify_branches.py -c skimmer/tests/modify_branches_skimmer.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op skimmer/tests/ -o picoAOD_modify_branches.yml -m metadata/datasets_HH4b.yml  -t
ls -R skimmer/
echo "############### Running skimmer test processor"
python runner.py -s -p skimmer/tests/modify_branches.py -c skimmer/tests/modify_branches_analysis.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op skimmer/tests/ -o picoAOD_modify_branches.coffea -m skimmer/tests/picoAOD_modify_branches.yml  -t
