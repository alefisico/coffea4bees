#!/bin/bash
source scripts/set_initial_variables.sh --output ${1:-"output/"} --do_proxy

OUTPUT_DIR="${DEFAULT_DIR}skimmer_test_job"
echo "############### Checking and creating output directory"
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

echo "############### Changing metadata"
if [[ $(hostname) = *fnal* ]]; then
    sed -e "s#base_path.*#base_path: \/srv\/output\/skimmer_test_job\/#" -e "s/\#max.*/maxchunks: 5/" -e "s/\#test.*/test_files: 1/" -e "s/2024_.*/tmp\//" skimmer/metadata/HH4b.yml > $OUTPUT_DIR/HH4b.yml
else
    sed -e "s#base_.*#base_path: \/builds\/${CI_PROJECT_PATH}\/output\/skimmer_test_job\/#" -e "s/\#max.*/maxchunks: 5/" -e "s/\#test.*/test_files: 1/" -e "s/2024_.*/tmp\//" -e "s/T3_US_FNALLPC/T3_CH_PSI/" skimmer/metadata/HH4b.yml > $OUTPUT_DIR/HH4b.yml
fi
cat $OUTPUT_DIR/HH4b.yml

#nanoAOD_file="root://cms-xrd-global.cern.ch//store/mc/RunIISummer20UL18NanoAODv9/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/92D0BDF3-91AE-514F-88B5-8F591450B8AD.root"
#sed -e "s#/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" metadata/datasets_HH4b.yml > $OUTPUT_DIR/datasets_HH4b.yml
#python runner.py -s -p skimmer/processor/skimmer_4b.py -c $OUTPUT_DIR/datasets_HH4b.yml -y UL18 -d TTToSemiLeptonic -op $OUTPUT_DIR -o picoaod_datasets_TTToSemiLeptonic_UL18.yml -m $OUTPUT_DIR/datasets_HH4b.yml  -t 

nanoAOD_file="root://cmseos.fnal.gov//store/mc/RunIISummer20UL18NanoAODv9/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/70000/3F95108D-84D2-CD4D-A0D2-324A7D15E691.root"
sed -e "s#/GluGluToHHTo4B_cHHH0_TuneCP5_PSWeights_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9.*#[ '${nanoAOD_file}' ]#" metadata/datasets_HH4b.yml > $OUTPUT_DIR/datasets_HH4b.yml
echo "############### Running test processor"
python runner.py -s -p skimmer/processor/skimmer_4b.py -c $OUTPUT_DIR/HH4b.yml -y UL18 -d GluGluToHHTo4B_cHHH0 -op $OUTPUT_DIR -o picoaod_datasets_GluGluToHHTo4B_cHHH0_UL18.yml -m $OUTPUT_DIR/datasets_HH4b.yml  -t 
ls -R $OUTPUT_DIR

if [ "$return_to_base" = true ]; then
    echo "############### Returning to base directory"
    cd ../
fi