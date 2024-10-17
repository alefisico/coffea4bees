echo "############### Including proxy"
if [ ! -f "${PWD}/proxy/x509_proxy" ]; then
    echo "Error: x509_proxy file not found!"
    exit 1
fi
export X509_USER_PROXY=${PWD}/proxy/x509_proxy

echo "############### Checking proxy"
voms-proxy-info

echo "############### Moving to python folder"
cd python/

echo "############### Checking and creating output/skimmer directory"
if [ ! -d "output/skimmer_basic_test_job" ]; then
    mkdir -p output/skimmer_basic_test_job
fi

echo "############### Changing metadata"
echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_.*#base_path: \/srv\/python\/output\/skimmer_basic_test_job\/#" skimmer/tests/modify_branches_skimmer.yml > output/skimmer_basic_test_job/modify_branches_skimmer.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/python\/output\/skimmer_basic_test_job\/#" skimmer/tests/modify_branches_skimmer.yml > output/skimmer_basic_test_job/modify_branches_skimmer.yml
fi
cat output/skimmer_basic_test_job/modify_branches_skimmer.yml

nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2810000/94DA5440-3B94-354B-A25B-78518A52D2D1.root"
sed -e "s#/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" metadata/datasets_HH4b.yml > output/skimmer_basic_test_job/datasets_HH4b.yml
cat output/skimmer_basic_test_job/datasets_HH4b.yml
echo "############### Skimming"
python runner.py -s -p skimmer/tests/modify_branches.py -c output/skimmer_basic_test_job/modify_branches_skimmer.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op output/skimmer_basic_test_job/ -o picoAOD_modify_branches.yml -m output/skimmer_basic_test_job/datasets_HH4b.yml -t --debug
ls -R skimmer/
echo "############### Checking skimmer output"
python metadata/merge_yaml_datasets.py -m output/skimmer_basic_test_job/datasets_HH4b.yml -f output/skimmer_basic_test_job/picoAOD_modify_branches.yml -o output/skimmer_basic_test_job/picoAOD_modify_branches.yml
python runner.py -p skimmer/tests/modify_branches.py -c skimmer/tests/modify_branches_analysis.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op output/skimmer_basic_test_job/ -o modify_branches.coffea -m output/skimmer_basic_test_job/picoAOD_modify_branches.yml -t --debug
