#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}/skimmer_basic_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_.*#base_path: \/srv\/output\/skimmer_basic_test_job\/#" skimmer/tests/modify_branches_skimmer.yml > ${OUTPUT_DIR}/modify_branches_skimmer.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/output\/skimmer_basic_test_job\/#" skimmer/tests/modify_branches_skimmer.yml > ${OUTPUT_DIR}/modify_branches_skimmer.yml
fi
cat ${OUTPUT_DIR}/modify_branches_skimmer.yml

nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2810000/94DA5440-3B94-354B-A25B-78518A52D2D1.root"
sed -e "s#/GluGluToHHTo4B_cHHH1_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" metadata/datasets_HH4b.yml > ${OUTPUT_DIR}/datasets_HH4b.yml
cat ${OUTPUT_DIR}/datasets_HH4b.yml

echo "############### Skimming"
python runner.py -s -p skimmer/tests/modify_branches.py -c ${OUTPUT_DIR}/modify_branches_skimmer.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op ${OUTPUT_DIR} -o picoAOD_modify_branches.yml -m ${OUTPUT_DIR}/datasets_HH4b.yml -t --debug
ls -R skimmer/

echo "############### Checking skimmer output"
python metadata/merge_yaml_datasets.py -m ${OUTPUT_DIR}/datasets_HH4b.yml -f ${OUTPUT_DIR}/picoAOD_modify_branches.yml -o ${OUTPUT_DIR}/picoAOD_modify_branches.yml
python runner.py -p skimmer/tests/modify_branches.py -c skimmer/tests/modify_branches_analysis.yml -y UL18 -d GluGluToHHTo4B_cHHH1 -op ${OUTPUT_DIR} -o modify_branches.coffea -m ${OUTPUT_DIR}/picoAOD_modify_branches.yml -t --debug

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi